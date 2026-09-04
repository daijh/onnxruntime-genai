import os
import json
import shutil
import subprocess
import sys
import argparse

import onnx

from builders.zimage_text_encoder import export_qwen3_text_encoder
from builders.zimage_helper_models import build_helper_models
from builders.zimage_safety_checker import build_safety_checker

# Maps the user-facing -p choice to the underlying builder.py `-p/--precision` value and
# whether MatMulNBits int4 weight quantization should be applied.
PRECISION_CONFIGS = {
    "f16": {"builder_precision": "fp16", "int4_quant": False},
    "f32": {"builder_precision": "fp32", "int4_quant": False},
    "f16_int4_quant": {"builder_precision": "int4", "int4_quant": True},
    "f32_int4_quant": {"builder_precision": "int4", "int4_quant": True},
}

# Small tokenizer files that `builder.py`'s AutoTokenizer.from_pretrained needs co-located
# with the weights. They live in the repo's sibling `tokenizer/` folder, not `text_encoder/`.
TOKENIZER_FILES = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")


def resolve_component_dir(input_path, component):
    # Accept either the Z-Image-Turbo repo root (containing a `<component>/` subfolder)
    # or the `<component>/` subfolder itself.
    candidate = os.path.join(input_path, component)
    if os.path.isfile(os.path.join(candidate, "config.json")):
        return candidate
    return input_path


def resolve_tokenizer_dir(input_path):
    # The tokenizer lives beside the text_encoder folder (repo_root/tokenizer), regardless of
    # whether input_path is the repo root or the text_encoder/ subfolder itself.
    text_encoder_dir = resolve_component_dir(input_path, "text_encoder")
    return os.path.join(os.path.dirname(os.path.normpath(text_encoder_dir)), "tokenizer")


def _copy_tokenizer_files(tokenizer_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    missing = []
    for fname in TOKENIZER_FILES:
        src = os.path.join(tokenizer_dir, fname)
        if os.path.isfile(src):
            _link_or_copy(src, os.path.join(dest_dir, fname))
        else:
            missing.append(fname)
    if missing:
        print(f"Warning: tokenizer files not found in {tokenizer_dir}: {missing}", file=sys.stderr)


def build_model(config):
    extra_options = str(config.get("extra_options", ""))
    precision_config = PRECISION_CONFIGS[config["precision"]]

    # Define the static (shared) options
    static_options = ["hf_remote=False"]
    if precision_config["int4_quant"]:
        static_options += [
                "block_size=32",
                "accuracy_level=4",
                "op_types_to_quantize=MatMul/Gather",
                ]
        # `-p int4` on the WebGPU EP defaults to float16 I/O (matching `f16_int4_quant`);
        # `use_webgpu_fp32` switches it to float32 I/O (`f32_int4_quant`).
        if config["precision"] == "f32_int4_quant":
            static_options.append("use_webgpu_fp32=true")

    # Get dynamic options from config and split them into a list
    dynamic_options = extra_options.split()

    # Merge them into one flat list of options
    all_options_list = static_options + dynamic_options

    # Construct the command list
    command = [
            "python", "builder.py",
            "-e", "webgpu",
            "-p", precision_config["builder_precision"],
            "--extra_options",
            *all_options_list,
            "-c", "tmp",
            "-i", config["input"],
            "-o", config["output"]
            ]

    print("\n" + " ".join(command))
    try:
        subprocess.run(command, check=True)
        print("\n######\nSuccess")
    except subprocess.CalledProcessError:
        print("\n######\nFail", file=sys.stderr)


def _link_or_copy(src, dst):
    # Prefer a cheap hardlink (same NTFS volume); fall back to a copy across volumes.
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# Text encoder filename suffix per precision -- f32/f32_int4_quant aren't supported yet
# (GroupQueryAttention under fp32 has never been verified for this Qwen3 config).
TEXT_ENCODER_PRECISIONS = {
    "f16": "f16",
    "f16_int4_quant": "q4f16",
}


def build_text_encoder(input_path, output_dir, precision):
    if precision not in TEXT_ENCODER_PRECISIONS:
        print(
            f"\n❌ ERROR: -m text_encoder / -m all does not support -p {precision}. "
            f"The text encoder builder only supports {sorted(TEXT_ENCODER_PRECISIONS)} "
            "(GroupQueryAttention under fp32 has not been verified for this Qwen3 config).",
            file=sys.stderr,
        )
        sys.exit(1)

    text_encoder_dir = resolve_component_dir(input_path, "text_encoder")
    if not os.path.isfile(os.path.join(text_encoder_dir, "config.json")):
        print(f"Could not find text_encoder/config.json under {input_path}", file=sys.stderr)
        return

    suffix = TEXT_ENCODER_PRECISIONS[precision]
    os.makedirs(output_dir, exist_ok=True)
    output_onnx = os.path.join(output_dir, f"text_encoder_model_{suffix}.onnx")
    export_qwen3_text_encoder(
        text_encoder_dir, output_onnx, f"text_encoder_model_{suffix}.onnx.data",
        quantize=(precision == "f16_int4_quant"),
    )
    print("\n######\nSuccess")


def build_vae_decoder(input_path, output_dir, precision):
    # precision: "f16" or "f32" only -- no int4/int8 quant, decomposed (fuse_group_norm=false)
    # GroupNorm, matching the WebNN bundle's own single unquantized vae_decoder_model_f16.onnx.
    # For finer control (int4/int8 mid-block quant, fuse_group_norm=true), call builder.py
    # directly -- see builders/ZIMAGE_VAE_USAGE.md.
    if precision not in ("f16", "f32"):
        raise ValueError(f"precision must be 'f16' or 'f32', got {precision!r}")

    vae_dir = resolve_component_dir(input_path, "vae")
    config_path = os.path.join(vae_dir, "config.json")
    if not os.path.isfile(config_path):
        print(f"Could not find vae/config.json under {input_path}", file=sys.stderr)
        return

    stage_root = os.path.join(output_dir, "_staging_vae")
    if os.path.isdir(stage_root):
        shutil.rmtree(stage_root, ignore_errors=True)

    builder_precision = "fp16" if precision == "f16" else "fp32"
    command = [
        "python", "builder.py",
        "-e", "webgpu",
        "-p", builder_precision,
        "--extra_options", "hf_remote=False",
        "-c", "tmp",
        "-i", vae_dir,
        "-o", stage_root,
    ]
    print("\n" + " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print("\n######\nFail (builder.py)", file=sys.stderr)
        return

    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, f"vae_decoder_model_{precision}.onnx")
    _rename_onnx_with_external_data(
        os.path.join(stage_root, "model.onnx"),
        final_path,
        f"vae_decoder_model_{precision}.onnx.data",
    )
    shutil.rmtree(stage_root, ignore_errors=True)
    print("\n######\nSuccess")


def _rename_onnx_with_external_data(onnx_path, new_onnx_path, new_external_data_name):
    # Rename an ONNX model + its external-data blob without reloading the (multi-GB) tensor
    # payload: load with load_external_data=False (graph proto only), patch the initializers'
    # `external_data` location strings, resave the small proto, then os.replace() (cheap,
    # same-volume rename) the actual data file.
    model = onnx.load(onnx_path, load_external_data=False)
    old_data_name = None
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key == "location":
                old_data_name = old_data_name or entry.value
                entry.value = new_external_data_name

    new_dir = os.path.dirname(new_onnx_path) or "."
    os.makedirs(new_dir, exist_ok=True)
    if old_data_name:
        old_data_path = os.path.join(os.path.dirname(onnx_path), old_data_name)
        new_data_path = os.path.join(new_dir, new_external_data_name)
        if os.path.exists(new_data_path):
            os.remove(new_data_path)
        os.replace(old_data_path, new_data_path)

    onnx.save(model, new_onnx_path)
    if os.path.abspath(onnx_path) != os.path.abspath(new_onnx_path):
        os.remove(onnx_path)


def helper_precision_from(precision):
    # Helper models have no int4-quantization concept (tiny non-weighted graphs); they only
    # differentiate float16 vs float32 I/O, matching the transformer precision's dtype half.
    return "f32" if precision in ("f32", "f32_int4_quant") else "f16"


# Transformer filename suffix per precision, matching the WebNN bundle's own naming convention
# (its int4/fp16 transformer is `transformer_model_q4f16.onnx`).
TRANSFORMER_FILENAME_SUFFIXES = {
    "f16": "f16",
    "f32": "f32",
    "f16_int4_quant": "q4f16",
    "f32_int4_quant": "q4f32",
}


def build_all(args):
    # One command builds everything this repo can build (transformer + text encoder + helper
    # models + safety checker + VAE decoder) into a single bundle-shaped directory:
    # onnx/*.onnx + tokenizer/*.
    if not args.safety_checker_checkpoint:
        print(
            "\n❌ ERROR: -m all requires --safety_checker_checkpoint.\n"
            "       Download it once with:\n"
            "         from huggingface_hub import snapshot_download\n"
            "         snapshot_download('CompVis/stable-diffusion-safety-checker', "
            "local_dir='path_to_safety_checker_folder')\n"
            "       then pass --safety_checker_checkpoint path_to_safety_checker_folder.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.precision not in TEXT_ENCODER_PRECISIONS:
        print(
            f"\n❌ ERROR: -m all does not support -p {args.precision} yet -- the text encoder "
            f"builder only supports {sorted(TEXT_ENCODER_PRECISIONS)} (GroupQueryAttention under "
            "fp32 has not been verified for this Qwen3 config). Use -p f16 or f16_int4_quant.",
            file=sys.stderr,
        )
        sys.exit(1)

    model_name = os.path.basename(os.path.normpath(args.input))
    bundle_dir = f"{model_name}-genai-wgpu-{args.precision}"
    onnx_dir = os.path.join(bundle_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    print("\n### [1/5] transformer ###")
    transformer_dir = resolve_component_dir(args.input, "transformer")
    build_model({"input": transformer_dir, "output": onnx_dir, "precision": args.precision})
    transformer_suffix = TRANSFORMER_FILENAME_SUFFIXES[args.precision]
    transformer_filename = f"transformer_model_{transformer_suffix}.onnx"
    _rename_onnx_with_external_data(
        os.path.join(onnx_dir, "model.onnx"),
        os.path.join(onnx_dir, transformer_filename),
        f"{transformer_filename}.data",
    )

    print("\n### [2/5] text_encoder ###")
    build_text_encoder(args.input, onnx_dir, args.precision)

    print("\n### [3/5] helper_models ###")
    helper_precision = helper_precision_from(args.precision)
    build_helper_models(onnx_dir, helper_precision)

    print("\n### [4/5] safety_checker ###")
    build_safety_checker(args.safety_checker_checkpoint, onnx_dir, helper_precision)

    print("\n### [5/5] vae_decoder ###")
    build_vae_decoder(args.input, onnx_dir, helper_precision)

    print("\n### tokenizer ###")
    _copy_tokenizer_files(resolve_tokenizer_dir(args.input), os.path.join(bundle_dir, "tokenizer"))

    text_encoder_suffix = TEXT_ENCODER_PRECISIONS[args.precision]
    run_cmd = (
        f"  python run_z_image_turbo.py {bundle_dir} "
        f"--transformer {bundle_dir}/onnx/{transformer_filename} "
        f"--text_encoder {bundle_dir}/onnx/text_encoder_model_{text_encoder_suffix}.onnx "
        f"--vae_decoder {bundle_dir}/onnx/vae_decoder_model_{helper_precision}.onnx "
        f"--scheduler_step {bundle_dir}/onnx/scheduler_step_model_{helper_precision}.onnx "
        f"--vae_pre_process {bundle_dir}/onnx/vae_pre_process_model_{helper_precision}.onnx "
        f"--sc_prep {bundle_dir}/onnx/sc_prep_model_{helper_precision}.onnx --safety_checker"
    )
    print(
        f"\n######\nDone. Bundle at {os.path.abspath(bundle_dir)}/ -- fully self-contained, no "
        f"WebNN bundle needed. Run the pipeline with, e.g.:\n{run_cmd}"
    )
    if args.precision == "f16_int4_quant":
        print(
            "\nSince this is the default f16_int4_quant precision, every filename above already "
            "matches the pipeline's built-in defaults, so the flags are optional -- this also "
            f"works:\n  python run_z_image_turbo.py {bundle_dir} --safety_checker"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="Input model folder: the Z-Image-Turbo repo root or a component subfolder",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=[
            "transformer", "text_encoder", "helper_models", "safety_checker", "vae_decoder", "all",
        ],
        default="transformer",
        help=(
            "Which Z-Image-Turbo component to build. 'all' builds transformer + text_encoder + "
            "helper_models + safety_checker + vae_decoder into one bundle-shaped directory "
            "(onnx/ + tokenizer/). Default: transformer."
        ),
    )
    parser.add_argument(
        "--safety_checker_checkpoint",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Local folder holding a pre-downloaded CompVis/stable-diffusion-safety-checker "
            "checkpoint (huggingface_hub.snapshot_download it once, same as the main model "
            "checkpoint). Required for -m safety_checker and -m all."
        ),
    )
    parser.add_argument(
        "-p",
        "--precision",
        choices=list(PRECISION_CONFIGS.keys()),
        default="f16_int4_quant",
        help=(
            "Precision to build: f16/f32 (unquantized WebGPU I/O dtype) or "
            "f16_int4_quant/f32_int4_quant (int4-quantized weights with float16/float32 "
            "WebGPU I/O). Default: f16_int4_quant. For -m helper_models/safety_checker/"
            "vae_decoder, only the f16-vs-f32 half applies (no int4 quantization). "
            "-m text_encoder/all support only f16 and f16_int4_quant."
        ),
    )
    args = parser.parse_args()

    model_name = os.path.basename(os.path.normpath(args.input))

    if args.model == "all":
        build_all(args)
    elif args.model == "text_encoder":
        output = f"{model_name}-text_encoder-genai-wgpu-{args.precision}"
        build_text_encoder(args.input, output, args.precision)
    elif args.model == "helper_models":
        helper_precision = helper_precision_from(args.precision)
        if args.precision not in ("f16", "f32"):
            print(
                f"Note: -m helper_models has no int4 quantization; building {helper_precision} "
                "I/O only."
            )
        output = f"{model_name}-helper_models-genai-wgpu-{helper_precision}"
        build_helper_models(output, helper_precision)
    elif args.model == "safety_checker":
        if not args.safety_checker_checkpoint:
            print(
                "\n❌ ERROR: -m safety_checker requires --safety_checker_checkpoint.\n"
                "       Download it once with:\n"
                "         from huggingface_hub import snapshot_download\n"
                "         snapshot_download('CompVis/stable-diffusion-safety-checker', "
                "local_dir='path_to_safety_checker_folder')",
                file=sys.stderr,
            )
            sys.exit(1)
        helper_precision = helper_precision_from(args.precision)
        if args.precision not in ("f16", "f32"):
            print(
                f"Note: -m safety_checker has no int4 quantization; building {helper_precision} "
                "I/O only."
            )
        output = f"{model_name}-safety_checker-genai-wgpu-{helper_precision}"
        build_safety_checker(args.safety_checker_checkpoint, output, helper_precision)
    elif args.model == "vae_decoder":
        helper_precision = helper_precision_from(args.precision)
        if args.precision not in ("f16", "f32"):
            print(
                f"Note: -m vae_decoder has no int4 quantization; building {helper_precision} "
                "I/O only. For int4/int8 mid-block quant or fuse_group_norm=true, call "
                "builder.py directly -- see builders/ZIMAGE_VAE_USAGE.md."
            )
        output = f"{model_name}-vae_decoder-genai-wgpu-{helper_precision}"
        build_vae_decoder(args.input, output, helper_precision)
    else:
        transformer_dir = resolve_component_dir(args.input, "transformer")
        output = f"{model_name}-transformer-genai-wgpu-{args.precision}"
        model_config = {
            "input": transformer_dir,
            "output": output,
            "precision": args.precision,
        }
        build_model(model_config)
