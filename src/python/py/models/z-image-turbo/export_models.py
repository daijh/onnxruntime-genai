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
    transformer    -> build_transformer.py   (implemented)
    vae_decoder    -> build_vae_decoder.py   (implemented)
    text_encoder   -> build_text_encoder.py  (not yet ported)
    helper_models  -> build_helper_models.py (not yet ported)
    safety_checker -> build_safety_checker.py (not yet ported)
    all            -> every component above, in one bundle directory
"""

import argparse
import os

import build_transformer
import build_vae_decoder

NOT_YET_PORTED = ("text_encoder", "helper_models", "safety_checker")


def get_args():
    parser = argparse.ArgumentParser(description="Export the Z-Image-Turbo pipeline to ONNX.")
    parser.add_argument("input_path", help="Path to the Z-Image-Turbo checkpoint (repo root or component subfolder).")
    parser.add_argument(
        "-o", "--output_dir", default=build_transformer.DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the ONNX model(s) to (under an `onnx/` subdir). Default: {build_transformer.DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-m", "--model", default="all",
        choices=["transformer", "vae_decoder", "text_encoder", "helper_models", "safety_checker", "all"],
        help="Which component to build. Default: all.",
    )
    parser.add_argument(
        "--extra_options", nargs="*", default=[],
        help="Extra key=value options passed through to the component's builder.",
    )
    return parser.parse_args()


def build_one(model, input_path, output_dir, extra_options):
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
    if model in NOT_YET_PORTED:
        raise NotImplementedError(
            f"-m {model} isn't ported to this standalone (pip-onnxruntime-genai-only) experiment yet. "
            f"Use ../build_z_image_turbo.py -m {model} for the version coupled to this repo's ../builders/ tree."
        )
    raise ValueError(f"Unknown -m/--model value: {model}")


def main():
    args = get_args()
    if args.model == "all":
        onnx_dir = args.output_dir
        for model in ("transformer", "vae_decoder", *NOT_YET_PORTED):
            try:
                onnx_dir = build_one(model, args.input_path, args.output_dir, args.extra_options)
            except NotImplementedError as e:
                print(f"Skipping {model}: {e}")
    else:
        onnx_dir = build_one(args.model, args.input_path, args.output_dir, args.extra_options)
    print(f"\nSuccess: {args.model} exported to {onnx_dir}")


if __name__ == "__main__":
    main()
