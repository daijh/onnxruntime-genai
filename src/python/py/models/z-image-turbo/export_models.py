# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
"""Main entry point for exporting the Z-Image-Turbo pipeline to standalone ONNX models.

Self-contained experiment (see requirements.txt): every build_*.py this dispatches to
depends only on the public onnxruntime-genai pip package, not on this repo's own
../builders/ source tree.

-m/--model selects which component to build; each maps to one build_*.py's `build()`:
    transformer    -> build_transformer.py     (implemented)
    vae_decoder    -> build_vae_decoder.py     (implemented)
    helper_models  -> build_helper_models.py   (implemented)
    safety_checker -> build_safety_checker.py  (implemented; needs its own separate
                       checkpoint, see --safety_checker_checkpoint)
    text_encoder   -> build_text_encoder.py    (not yet ported)
    all            -> every component above, in one bundle directory (safety_checker is
                       skipped, with a message, if --safety_checker_checkpoint isn't given)

Also copies the checkpoint's tokenizer files into `<output_dir>/tokenizer/` (see
../build_z_image_turbo.py, which does the same) so the exported directory is self-contained
and can be pointed at directly, e.g. by run_z_image_turbo.py.
"""

import argparse
import os
import shutil
import sys

import build_helper_models
import build_safety_checker
import build_transformer
import build_vae_decoder

NOT_YET_PORTED = ("text_encoder",)

# Small tokenizer files that AutoTokenizer.from_pretrained needs; they live in the checkpoint's
# sibling `tokenizer/` folder, not `text_encoder/`.
TOKENIZER_FILES = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")


def get_args():
    parser = argparse.ArgumentParser(description="Export the Z-Image-Turbo pipeline to ONNX.")
    parser.add_argument("input_path", help="Path to the Z-Image-Turbo checkpoint (repo root or component subfolder).")
    parser.add_argument(
        "-o", "--output_dir", default=build_transformer.DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the ONNX model(s) to (under an `onnx/` subdir). Default: {build_transformer.DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-m", "--model", default="all",
        choices=["transformer", "vae_decoder", "helper_models", "text_encoder", "safety_checker", "all"],
        help="Which component to build. Default: all.",
    )
    parser.add_argument(
        "--extra_options", nargs="*", default=[],
        help="Extra key=value options passed through to the component's builder.",
    )
    parser.add_argument(
        "--safety_checker_checkpoint", default="",
        help="Path to a local CompVis/stable-diffusion-safety-checker-compatible checkpoint "
        "(e.g. via huggingface_hub.snapshot_download('CompVis/stable-diffusion-safety-checker', "
        "local_dir=...)). This is a separate download from input_path. Required for "
        "-m safety_checker; for -m all, safety_checker is skipped if this isn't given.",
    )
    return parser.parse_args()


def build_one(model, input_path, output_dir, extra_options, safety_checker_checkpoint=""):
    if model == "transformer":
        transformer_input = input_path
        if os.path.isdir(os.path.join(input_path, "transformer")):
            transformer_input = os.path.join(input_path, "transformer")
        return build_transformer.build(
            transformer_input, output_dir, extra_options=build_transformer.parse_extra_options(extra_options)
        )
    if model == "vae_decoder":
        vae_input = input_path
        if os.path.isdir(os.path.join(input_path, "vae")):
            vae_input = os.path.join(input_path, "vae")
        return build_vae_decoder.build(
            vae_input, output_dir, extra_options=build_vae_decoder.parse_extra_options(extra_options)
        )
    if model == "helper_models":
        return build_helper_models.build(
            input_path, output_dir, extra_options=build_helper_models.parse_extra_options(extra_options)
        )
    if model == "safety_checker":
        if not safety_checker_checkpoint:
            raise NotImplementedError(
                "-m safety_checker requires --safety_checker_checkpoint (a separate local "
                "CompVis/stable-diffusion-safety-checker-compatible checkpoint, not input_path)."
            )
        return build_safety_checker.build(
            safety_checker_checkpoint, output_dir, extra_options=build_safety_checker.parse_extra_options(extra_options)
        )
    if model in NOT_YET_PORTED:
        raise NotImplementedError(
            f"-m {model} isn't ported to this standalone (pip-onnxruntime-genai-only) experiment yet. "
            f"Use ../build_z_image_turbo.py -m {model} for the version coupled to this repo's ../builders/ tree."
        )
    raise ValueError(f"Unknown -m/--model value: {model}")


def resolve_tokenizer_dir(input_path):
    # The tokenizer lives beside the text_encoder folder (repo_root/tokenizer), regardless of
    # whether input_path is the repo root or a component subfolder itself.
    text_encoder_dir = input_path
    if os.path.isdir(os.path.join(input_path, "text_encoder")):
        text_encoder_dir = os.path.join(input_path, "text_encoder")
    return os.path.join(os.path.dirname(os.path.normpath(text_encoder_dir)), "tokenizer")


def _link_or_copy(src, dst):
    # Prefer a cheap hardlink (same volume); fall back to a copy across volumes.
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_tokenizer(input_path, output_dir):
    tokenizer_dir = resolve_tokenizer_dir(input_path)
    if not os.path.isdir(tokenizer_dir):
        print(f"Skipping tokenizer copy: {tokenizer_dir} not found", file=sys.stderr)
        return

    dest_dir = os.path.join(output_dir, "tokenizer")
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
    else:
        print(f"Copied tokenizer to {dest_dir}")


def main():
    args = get_args()
    if args.model == "all":
        onnx_dir = args.output_dir
        for model in ("transformer", "vae_decoder", "helper_models", "safety_checker", *NOT_YET_PORTED):
            try:
                onnx_dir = build_one(
                    model, args.input_path, args.output_dir, args.extra_options, args.safety_checker_checkpoint
                )
            except NotImplementedError as e:
                print(f"Skipping {model}: {e}")
    else:
        onnx_dir = build_one(
            args.model, args.input_path, args.output_dir, args.extra_options, args.safety_checker_checkpoint
        )

    copy_tokenizer(args.input_path, args.output_dir)
    print(f"\nSuccess: {args.model} exported to {onnx_dir}")


if __name__ == "__main__":
    main()
