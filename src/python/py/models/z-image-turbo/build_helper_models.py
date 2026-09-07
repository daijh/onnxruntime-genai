# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
"""Build the scheduler_step / vae_pre_process / sc_prep helper ONNX graphs for the self-built
Z-Image-Turbo pipeline, in either float16 or float32.

Self-contained port of ../builders/zimage_helper_models.py: unlike build_transformer.py /
build_vae_decoder.py, this one never depended on onnxruntime-genai's `Model` base class in
the first place (these graphs are hand-authored, no pretrained weights, no checkpoint input
at all) -- only `torch`, `onnx`, `onnxruntime`, `onnxconverter_common`, and `numpy`.

These mirror the tiny helper graphs the WebNN JS demo generates to keep intermediate tensors
GPU-resident, but shaped to match the self-built transformer's `[1, 16, H, W]`
hidden_states/sample convention instead of the WebNN bundle's `[*, 16, 1, H, W]` (no
`num_frames` axis here), and with a genuinely float16 I/O boundary in the f16 variant (not just
float16-internal-with-float32-boundary).
"""

import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_CHANNELS = 16

VAE_SCALING_FACTOR = 0.3611
VAE_SHIFT_FACTOR = 0.1159

# Standard OpenAI CLIP preprocessing constants.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PRECISION_CONFIGS = {"f16": {}, "f32": {}}

DEFAULT_OUTPUT_DIR = "z-image-turbo-onnx"


def parse_extra_options(pairs):
    """Parse `key=value` strings (as passed via `--extra_options`) into a dict."""
    extra_options = {}
    for kv_str in pairs or []:
        key, _, value = kv_str.partition("=")
        extra_options[key.strip()] = value.strip()
    return extra_options


# ======================================================================
# Model 1 - Scheduler Step (one flow-matching Euler step + sigma calc)
# ======================================================================
class SchedulerStepModel(nn.Module):
    """
    Performs one Euler step, recomputing sigma/sigma_next internally from
    step_info=[step, num_steps] (shift=3 schedule -- matches the deployed WebNN pipeline).

    Inputs:  noise_pred [1, 16, H, W], latents [1, 16, H, W], step_info [2]
    Output:  latents_out [1, 16, H, W]
    """

    def __init__(self):
        super().__init__()
        timesteps = torch.linspace(1, 1000, 1000, dtype=torch.float32).flip(0)
        s = timesteps / 1000.0
        shift = 3.0
        sigmas = (shift * s) / (1.0 + (shift - 1.0) * s)
        self.register_buffer("sigma_max", sigmas[0])
        self.register_buffer("sigma_min", sigmas[-1])
        self.register_buffer("shift", torch.tensor(shift, dtype=torch.float32))

    def forward(self, noise_pred, latents, step_info):
        step, num_steps = step_info[0], step_info[1]

        t_start = self.sigma_max * 1000.0
        t_end = self.sigma_min * 1000.0
        denom = num_steps - 1.0
        denom = torch.where(denom == 0.0, torch.tensor(1.0), denom)
        step_delta = (t_end - t_start) / denom

        t_curr = t_start + step * step_delta
        t_next = t_start + (step + 1.0) * step_delta

        s_curr = t_curr / 1000.0
        s_next = t_next / 1000.0
        sigma_curr = (self.shift * s_curr) / (1.0 + (self.shift - 1.0) * s_curr)
        sigma_next_raw = (self.shift * s_next) / (1.0 + (self.shift - 1.0) * s_next)

        is_last = step >= (num_steps - 1.0)
        sigma_next = torch.where(is_last, torch.tensor(0.0), sigma_next_raw)

        dt = sigma_next - sigma_curr
        return latents - dt * noise_pred


def numpy_scheduler_step(noise_pred, latents, step, num_steps):
    timesteps = np.linspace(1, 1000, 1000, dtype=np.float32)[::-1]
    s = timesteps / 1000.0
    shift = 3.0
    sigmas = (shift * s) / (1.0 + (shift - 1.0) * s)
    sigma_max, sigma_min = sigmas[0], sigmas[-1]
    t_start, t_end = sigma_max * 1000.0, sigma_min * 1000.0
    denom = max(num_steps - 1.0, 1.0)
    step_delta = (t_end - t_start) / denom
    t_curr = t_start + step * step_delta
    t_next = t_start + (step + 1) * step_delta
    s_curr, s_next = t_curr / 1000.0, t_next / 1000.0
    sigma_curr = (shift * s_curr) / (1.0 + (shift - 1.0) * s_curr)
    if step >= num_steps - 1:
        sigma_next = 0.0
    else:
        sigma_next = (shift * s_next) / (1.0 + (shift - 1.0) * s_next)
    dt = sigma_next - sigma_curr
    return latents - dt * noise_pred


# ======================================================================
# Model 2 - VAE Pre-Process (scale/shift, no frame-axis squeeze)
# ======================================================================
class VaePreProcessModel(nn.Module):
    """
    VAE scale/shift. Our latents have no frame axis to squeeze (already [1, 16, H, W]),
    unlike the WebNN bundle's `squeeze(axis=2) / scaling_factor + shift_factor`.

    Input:  latents [1, 16, H, W]
    Output: scaled_latents [1, 16, H, W]
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("scaling_factor", torch.tensor(VAE_SCALING_FACTOR, dtype=torch.float32))
        self.register_buffer("shift_factor", torch.tensor(VAE_SHIFT_FACTOR, dtype=torch.float32))

    def forward(self, latents):
        return latents / self.scaling_factor + self.shift_factor


# ======================================================================
# Model 3 - sc_prep (resize + CLIP normalize, for the safety checker)
# ======================================================================
class ScPrepModel(nn.Module):
    """
    Resize the raw VAE-decoded image ([-1, 1] range) to CLIP's 224x224 input and normalize with
    the standard OpenAI CLIP mean/std. Folds the VAE's [-1,1]->[0,1] denorm into the per-channel
    affine (scale = 0.5/std, offset = (0.5-mean)/std) -- reverse-engineered from the deployed
    bundle's sc_prep_model_f16.onnx (Resize mode=linear, coordinate_transformation_mode=
    asymmetric, then Mul+Add by per-channel constants matching these exactly to 4 decimals).

    Input:  sample [1, 3, H, W]      (VAE decoder output, [-1, 1] range)
    Output: clip_input [1, 3, 224, 224]
    """

    def __init__(self):
        super().__init__()
        std = torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
        mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("scale", 0.5 / std)
        self.register_buffer("offset", (0.5 - mean) / std)

    def forward(self, sample):
        resized = F.interpolate(sample, size=(224, 224), mode="bilinear", align_corners=False)
        return resized * self.scale + self.offset


def numpy_resize_asymmetric_bilinear(img, out_h, out_w):
    # img: [C, H, W] float32. "asymmetric" coordinate transform: in_coord = out_coord * (in/out),
    # no half-pixel centering -- matches the deployed sc_prep model's Resize attribute.
    c, h, w = img.shape
    scale_h, scale_w = h / out_h, w / out_w
    ys = np.arange(out_h) * scale_h
    xs = np.arange(out_w) * scale_w
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wy = (ys - y0).astype(np.float32)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wx = (xs - x0).astype(np.float32)

    top = img[:, y0][:, :, x0] * (1 - wx)[None, None, :] + img[:, y0][:, :, x1] * wx[None, None, :]
    bot = img[:, y1][:, :, x0] * (1 - wx)[None, None, :] + img[:, y1][:, :, x1] * wx[None, None, :]
    return (top * (1 - wy)[None, :, None] + bot * wy[None, :, None]).astype(np.float32)


def numpy_sc_prep(sample):
    # sample: [1, 3, H, W] float32
    resized = numpy_resize_asymmetric_bilinear(sample[0], 224, 224)[None, ...]
    std = np.array(CLIP_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    mean = np.array(CLIP_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    return resized * (0.5 / std) + (0.5 - mean) / std


# ======================================================================
# Export helpers
# ======================================================================
def _force_dynamic_dim_names(onnx_model, dynamic_axes):
    # Some torch/opset combinations don't fully honor dynamic_axes on export; patch symbolic
    # dim names in directly as a safety net (mirrors the WebNN reference generator's approach).
    tensors = {t.name: t for t in list(onnx_model.graph.input) + list(onnx_model.graph.output)}
    for name, axes in dynamic_axes.items():
        tensor = tensors.get(name)
        if tensor is None:
            continue
        dims = tensor.type.tensor_type.shape.dim
        for axis, dim_name in axes.items():
            if axis < len(dims):
                dims[axis].dim_param = dim_name
                if dims[axis].HasField("dim_value"):
                    dims[axis].ClearField("dim_value")


def _force_resize_asymmetric(onnx_model):
    # torch's bilinear interpolate (align_corners=False) exports as pytorch_half_pixel; force it
    # to asymmetric to match the deployed sc_prep model exactly (see ScPrepModel docstring).
    for node in onnx_model.graph.node:
        if node.op_type != "Resize":
            continue
        for attr in node.attribute:
            if attr.name == "coordinate_transformation_mode":
                attr.s = b"asymmetric"
                break
        else:
            node.attribute.append(onnx.helper.make_attribute("coordinate_transformation_mode", "asymmetric"))


def export_model(model, dummy_inputs, input_names, output_names, dynamic_axes, f32_path, opset=17):
    os.makedirs(os.path.dirname(f32_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_inputs,
        f32_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(f32_path)
    _force_dynamic_dim_names(onnx_model, dynamic_axes)
    _force_resize_asymmetric(onnx_model)
    onnx.save(onnx_model, f32_path)
    print(f"  wrote {f32_path} ({os.path.getsize(f32_path) / 1024:.1f} KB)")


def _fix_cast_node_types(onnx_model):
    # onnxconverter_common's float16 converter can rewrite a Cast node's declared OUTPUT type
    # (value_info) to float16 without updating the node's own `to` attribute, leaving a
    # Cast(to=FLOAT) node whose declared output is FLOAT16 -- onnxruntime rejects that at load
    # ("Type Error: ... does not match expected type"). Patch `to` to match the declared type.
    vi = {v.name: v for v in onnx_model.graph.value_info}
    for node in onnx_model.graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                out_type = vi.get(node.output[0])
                if out_type and out_type.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
                    attr.i = onnx.TensorProto.FLOAT16


def convert_to_f16(f32_path, f16_path):
    from onnxconverter_common import float16

    onnx_model = onnx.load(f32_path)
    # Resize is in onnxconverter_common's default op_block_list -- it gets left in fp32 with
    # Cast nodes wrapped around it -- because the ONNX spec hardcodes its `scales` input to
    # tensor(float) regardless of the data dtype. Our graphs only ever drive Resize via `sizes`
    # (int64) with empty roi/scales, and ORT's Resize kernel natively supports float16 X/Y, so
    # un-block just Resize to get a genuinely all-fp16 graph instead of two wasted Casts.
    op_block_list = [op for op in float16.DEFAULT_OP_BLOCK_LIST if op != "Resize"]
    onnx_model_f16 = float16.convert_float_to_float16(
        onnx_model, keep_io_types=False, op_block_list=op_block_list
    )
    _fix_cast_node_types(onnx_model_f16)
    onnx.save(onnx_model_f16, f16_path)
    print(f"  wrote {f16_path} ({os.path.getsize(f16_path) / 1024:.1f} KB)")


def verify_model(path, feed_dict, label):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outputs = sess.run(None, feed_dict)
    for meta, arr in zip(sess.get_outputs(), outputs):
        print(
            f"    [{label}] output '{meta.name}': shape={arr.shape}, dtype={arr.dtype}, "
            f"min={float(arr.min()):.4f}, max={float(arr.max()):.4f}, mean={float(arr.mean()):.4f}"
        )
    return outputs


def build_one(name, model, dummy_inputs, input_names, output_names, dynamic_axes, out_dir, precision, verify_fn):
    # precision is exactly "f16" or "f32". The f32 export is always produced first (it's either
    # the final artifact or the source for the f16 conversion), then removed if not requested.
    f32_path = os.path.join(out_dir, f"{name}_f32.onnx")
    export_model(model, dummy_inputs, input_names, output_names, dynamic_axes, f32_path)
    if precision == "f32":
        verify_fn(f32_path, "f32")
        return

    f16_path = os.path.join(out_dir, f"{name}_f16.onnx")
    convert_to_f16(f32_path, f16_path)
    verify_fn(f16_path, "f16")
    os.remove(f32_path)
    print(f"  (removed intermediate {f32_path}; only f16 was requested)")


# ======================================================================
# Standalone build driver
# ======================================================================
def build(input_path, output_dir, precision="f16", extra_options=None):
    """Build scheduler_step_model / vae_pre_process_model / sc_prep_model into `output_dir`/onnx.

    `input_path` is unused: these graphs are hand-authored (no pretrained weights, no
    checkpoint), unlike build_transformer.py/build_vae_decoder.py -- kept as a parameter only
    so export_models.py can dispatch to every component's `build()` uniformly.
    """
    del input_path
    if precision not in PRECISION_CONFIGS:
        raise ValueError(f"Unknown precision '{precision}'; choose from {sorted(PRECISION_CONFIGS)}")

    extra_options = dict(extra_options or {})
    height = int(extra_options.get("height", 512))
    width = int(extra_options.get("width", 512))
    num_inference_steps = int(extra_options.get("num_inference_steps", 8))

    # All .onnx output goes under a shared `onnx/` subdir of `output_dir`, so multiple
    # components can land side by side in one bundle.
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    latent_h, latent_w = height // 8, width // 8

    print("=" * 60)
    print("Generating Z-Image-Turbo helper models (self-built shape convention)")
    print(f"output_dir={onnx_dir} precision={precision}")
    print("=" * 60)

    # ── 1. scheduler_step ────────────────────────────────────────────────
    print("\n[1/3] scheduler_step_model")
    np_noise = np.random.randn(1, LATENT_CHANNELS, 24, 32).astype(np.float32)
    np_lat = np.random.randn(1, LATENT_CHANNELS, 24, 32).astype(np.float32)
    step_val, num_steps_val = 2, num_inference_steps

    def verify_scheduler_step(path, label):
        outputs = verify_model(
            path,
            {
                "noise_pred": np_noise.astype(np.float16 if label == "f16" else np.float32),
                "latents": np_lat.astype(np.float16 if label == "f16" else np.float32),
                "step_info": np.array([step_val, num_steps_val], dtype=np.float16 if label == "f16" else np.float32),
            },
            label,
        )
        ref = numpy_scheduler_step(np_noise, np_lat, step_val, num_steps_val)
        diff = np.max(np.abs(outputs[0].astype(np.float32) - ref))
        print(f"    [{label}] max abs diff vs numpy reference: {diff:.6e}")

    build_one(
        "scheduler_step_model",
        SchedulerStepModel().eval(),
        (
            torch.randn(1, LATENT_CHANNELS, latent_h, latent_w),
            torch.randn(1, LATENT_CHANNELS, latent_h, latent_w),
            torch.tensor([0.0, float(num_inference_steps)], dtype=torch.float32),
        ),
        ["noise_pred", "latents", "step_info"],
        ["latents_out"],
        # batch is fixed at 1 across the whole pipeline (the transformer hardcodes batch=1);
        # leave axis 0 out of dynamic_axes so it stays a static 1 (dummy inputs are batch=1).
        {
            "noise_pred": {2: "height", 3: "width"},
            "latents": {2: "height", 3: "width"},
            "step_info": {},
            "latents_out": {2: "height", 3: "width"},
        },
        onnx_dir, precision, verify_scheduler_step,
    )

    # ── 2. vae_pre_process ───────────────────────────────────────────────
    print("\n[2/3] vae_pre_process_model")
    np_vae_lat = np.random.randn(1, LATENT_CHANNELS, 24, 32).astype(np.float32)

    def verify_vae_pre_process(path, label):
        outputs = verify_model(
            path, {"latents": np_vae_lat.astype(np.float16 if label == "f16" else np.float32)}, label,
        )
        ref = np_vae_lat / VAE_SCALING_FACTOR + VAE_SHIFT_FACTOR
        diff = np.max(np.abs(outputs[0].astype(np.float32) - ref))
        print(f"    [{label}] max abs diff vs numpy reference: {diff:.6e}")

    build_one(
        "vae_pre_process_model",
        VaePreProcessModel().eval(),
        (torch.randn(1, LATENT_CHANNELS, latent_h, latent_w),),
        ["latents"], ["scaled_latents"],
        {
            "latents": {2: "height", 3: "width"},
            "scaled_latents": {2: "height", 3: "width"},
        },
        onnx_dir, precision, verify_vae_pre_process,
    )

    # ── 3. sc_prep ───────────────────────────────────────────────────────
    print("\n[3/3] sc_prep_model")
    np_sample = np.random.uniform(-1.0, 1.0, size=(1, 3, 48, 64)).astype(np.float32)

    def verify_sc_prep(path, label):
        outputs = verify_model(
            path, {"sample": np_sample.astype(np.float16 if label == "f16" else np.float32)}, label,
        )
        assert outputs[0].shape == (1, 3, 224, 224), f"unexpected sc_prep output shape {outputs[0].shape}"
        ref = numpy_sc_prep(np_sample)
        diff = np.max(np.abs(outputs[0].astype(np.float32) - ref))
        print(f"    [{label}] max abs diff vs numpy reference: {diff:.6e}")

    build_one(
        "sc_prep_model",
        ScPrepModel().eval(),
        (torch.randn(1, 3, height, width),),
        ["sample"], ["clip_input"],
        {
            "sample": {2: "height", 3: "width"},
            "clip_input": {},
        },
        onnx_dir, precision, verify_sc_prep,
    )

    print("\n" + "=" * 60)
    print(f"Done! Models saved to {os.path.abspath(onnx_dir)}/")
    print("=" * 60)
    return onnx_dir


def get_args():
    parser = argparse.ArgumentParser(description="Export the Z-Image-Turbo helper models to ONNX.")
    parser.add_argument(
        "-o", "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the ONNX models to (under an `onnx/` subdir). Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-p", "--precision", default="f16", choices=sorted(PRECISION_CONFIGS),
        help="Output precision. Default: f16.",
    )
    parser.add_argument(
        "--extra_options", nargs="*", default=[],
        help="Extra key=value options, e.g. height=512 width=512 num_inference_steps=8.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    onnx_dir = build(None, args.output_dir, args.precision, parse_extra_options(args.extra_options))
    print(f"\nSuccess: helper_models exported to {onnx_dir}")


if __name__ == "__main__":
    main()
