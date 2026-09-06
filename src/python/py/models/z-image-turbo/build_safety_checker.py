# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
"""Standalone exporter for the Z-Image-Turbo NSFW safety checker.

Self-contained port of ../builders/zimage_safety_checker.py: unlike build_transformer.py /
build_vae_decoder.py, this component was never coupled to onnxruntime-genai's `Model` base
class in the first place -- it's a real pretrained CLIP ViT-L/14 vision classifier
(cosine-distance threshold check against 17 "concept" and 3 "special care" reference
embeddings), exported via `diffusers`' `StableDiffusionSafetyChecker`, not authored from
scratch like the transformer/VAE. Only `torch`, `diffusers`, `onnx`, `onnxruntime`, and
`onnxconverter_common` (via build_helper_models.convert_to_f16, reused rather than
duplicated) are needed.

`diffusers`' `StableDiffusionSafetyChecker` already ships a `forward_onnx(clip_input, images)`
method written for ONNX export; `SafetyCheckerOnnxWrapper` below wraps it with the `images`
masking input/output trimmed off, so the graph is just `clip_input -> has_nsfw_concepts` --
matching the deployed WebNN bundle's `safety_checker_model_f16.onnx` (confirmed by comparing
initializer names/shapes: e.g. `vision_model.vision_model.embeddings.patch_embedding.weight
[1024,3,14,14]`, `concept_embeds [17,768]`, `special_care_embeds [3,768]` line up exactly with
this class's `__init__`).

The checkpoint is a *separate* download from the Z-Image-Turbo checkpoint itself -- see
`--safety_checker_checkpoint` in export_models.py, or pass a local
`CompVis/stable-diffusion-safety-checker`-compatible folder directly to this script.
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
    cosine_distance,
)

from build_helper_models import convert_to_f16

DEFAULT_OUTPUT_DIR = "z-image-turbo-onnx"

PRECISION_CONFIGS = {"f16": {}, "f32": {}}


def parse_extra_options(pairs):
    """Parse `key=value` strings (as passed via `--extra_options`) into a dict."""
    extra_options = {}
    for kv_str in pairs or []:
        key, _, value = kv_str.partition("=")
        extra_options[key.strip()] = value.strip()
    return extra_options


class SafetyCheckerOnnxWrapper(nn.Module):
    """`StableDiffusionSafetyChecker.forward_onnx` minus the `images` masking I/O."""

    def __init__(self, safety_checker):
        super().__init__()
        self.vision_model = safety_checker.vision_model
        self.visual_projection = safety_checker.visual_projection
        self.concept_embeds = safety_checker.concept_embeds
        self.special_care_embeds = safety_checker.special_care_embeds
        self.concept_embeds_weights = safety_checker.concept_embeds_weights
        self.special_care_embeds_weights = safety_checker.special_care_embeds_weights

    def forward(self, clip_input):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        special_scores = special_cos_dist - self.special_care_embeds_weights
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        return has_nsfw_concepts


def build(input_path, output_dir, precision="f16", extra_options=None, opset=17):
    """Export the safety checker to `<output_dir>/onnx/safety_checker_model_<precision>.onnx`.

    Args:
        input_path: local folder holding a pre-downloaded `CompVis/stable-diffusion-safety-
            checker`-compatible checkpoint (config.json + weights), e.g. via
            `huggingface_hub.snapshot_download`.
        output_dir: directory to write into (under an `onnx/` subdir, like the other
            build_*.py scripts).
        precision: "f16" (default) or "f32".
    """
    if precision not in PRECISION_CONFIGS:
        raise ValueError(f"Unknown precision '{precision}'; choose from {sorted(PRECISION_CONFIGS)}")

    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    print(f"Loading safety checker checkpoint from {input_path}")
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(input_path)
    safety_checker.eval()
    wrapper = SafetyCheckerOnnxWrapper(safety_checker).eval()

    f32_path = os.path.join(onnx_dir, "safety_checker_model_f32.onnx")
    dummy_clip_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        wrapper,
        (dummy_clip_input,),
        f32_path,
        input_names=["clip_input"],
        output_names=["has_nsfw_concepts"],
        dynamic_axes={
            "clip_input": {0: "batch"},
            "has_nsfw_concepts": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  wrote {f32_path} ({os.path.getsize(f32_path) / (1024 * 1024):.1f} MB)")

    if precision == "f32":
        final_path = f32_path
    else:
        final_path = os.path.join(onnx_dir, "safety_checker_model_f16.onnx")
        convert_to_f16(f32_path, final_path)
        os.remove(f32_path)
        print(f"  (removed intermediate {f32_path}; only f16 was requested)")
        print(f"  wrote {final_path} ({os.path.getsize(final_path) / (1024 * 1024):.1f} MB)")

    sess = ort.InferenceSession(final_path, providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print(f"input: {i}")
    for o in sess.get_outputs():
        print(f"output: {o}")

    dtype = np.float16 if precision == "f16" else np.float32
    sample = np.random.randn(2, 3, 224, 224).astype(dtype)
    outputs = sess.run(None, {"clip_input": sample})
    print(
        f"  sanity run output 'has_nsfw_concepts': shape={outputs[0].shape}, "
        f"dtype={outputs[0].dtype}, values={outputs[0]}"
    )

    return onnx_dir


def get_args():
    parser = argparse.ArgumentParser(description="Export the Z-Image-Turbo NSFW safety checker to ONNX.")
    parser.add_argument(
        "input_path",
        help="Path to a local CompVis/stable-diffusion-safety-checker-compatible checkpoint "
        "(config.json + weights), e.g. via huggingface_hub.snapshot_download('CompVis/"
        "stable-diffusion-safety-checker', local_dir=...).",
    )
    parser.add_argument(
        "-o", "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the ONNX model to (under an `onnx/` subdir). Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-p", "--precision", default="f16", choices=sorted(PRECISION_CONFIGS),
        help="Output precision. Default: f16.",
    )
    parser.add_argument(
        "--extra_options", nargs="*", default=[],
        help="Extra key=value options (currently unused; kept for CLI consistency with the "
        "other build_*.py scripts).",
    )
    return parser.parse_args()


def main():
    args = get_args()
    onnx_dir = build(args.input_path, args.output_dir, args.precision, parse_extra_options(args.extra_options))
    print(f"\nSuccess: safety_checker exported to {onnx_dir}")


if __name__ == "__main__":
    main()
