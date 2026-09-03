# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Export the NSFW safety checker to ONNX from the pretrained
`CompVis/stable-diffusion-safety-checker` checkpoint via `diffusers`.

Unlike the transformer/text encoder/helper models, this component isn't authored from scratch --
it's a real pretrained CLIP ViT-L/14 vision classifier (cosine-distance threshold check against
17 "concept" and 3 "special care" reference embeddings). `diffusers`'
`StableDiffusionSafetyChecker` already ships a `forward_onnx(clip_input, images)` method written
for ONNX export; this module wraps it with the `images` masking input/output trimmed off, so the
graph is just `clip_input -> has_nsfw_concepts` -- matching the deployed WebNN bundle's
`safety_checker_model_f16.onnx` (confirmed by comparing initializer names/shapes: e.g.
`vision_model.vision_model.embeddings.patch_embedding.weight [1024,3,14,14]`, `concept_embeds
[17,768]`, `special_care_embeds [3,768]` line up exactly with this class's `__init__`).
"""

import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
    cosine_distance,
)

from builders.zimage_helper_models import convert_to_f16


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


def build_safety_checker(checkpoint_dir, output_dir, precision, opset=17):
    """Export the safety checker to output_dir/safety_checker_model_<precision>.onnx.

    Args:
        checkpoint_dir: local folder holding a pre-downloaded `CompVis/stable-diffusion-safety-
            checker`-compatible checkpoint (config.json + weights), e.g. via
            `huggingface_hub.snapshot_download`.
        precision: "f16" or "f32".
    """
    if precision not in ("f16", "f32"):
        raise ValueError(f"precision must be 'f16' or 'f32', got {precision!r}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading safety checker checkpoint from {checkpoint_dir}")
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(checkpoint_dir)
    safety_checker.eval()
    wrapper = SafetyCheckerOnnxWrapper(safety_checker).eval()

    f32_path = os.path.join(output_dir, "safety_checker_model_f32.onnx")
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
        final_path = os.path.join(output_dir, "safety_checker_model_f16.onnx")
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

    return final_path
