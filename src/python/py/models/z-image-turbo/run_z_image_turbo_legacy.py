"""Reference driver for the legacy WebNN Z-Image-Turbo bundle in `z-image-turbo-webnn/`
(straight HF download of the `webnn/Z-Image-Turbo` bundle -- not produced by export_models.py).

Clone of `run_z_image_turbo.py` (see that file for the standalone-export version) kept as
close to identical as possible; the only real differences are forced by this bundle's own
shape/dtype conventions, introspected from the actual `.onnx` files:

- Every model here is float32 I/O (despite the "q4f16"/"f16" filename suffixes, which refer to
  the internal weight quantization, not the graph boundary dtype). `input_dtype()` already
  auto-detects this from each session, so no hardcoded float16 assumption needed changing.
- `latents` carries an explicit num_frames=1 axis: `[1, 16, 1, H, W]` (rank 5), not `[1, 16, H, W]`
  -- so the initial noise tensor is created with that extra axis (see `run()`), and the
  transformer's raw `noise_pred` output is `[16, 1, H, W]` (no batch axis at all). Every other
  shape (`scheduler_step`'s `latents` in/out, `vae_pre_process`'s `latents` in) already matches
  what each model naturally produces/expects, so no further reshaping is needed anywhere --
  unlike the older `../run_z_image_turbo.py`, which had to support switching between this bundle
  and self-built dev models with a different shape convention, this driver only ever targets one
  fixed convention.
- The transformer has its own internal attention-mask/padding logic (no separate mask input, but
  it accepts any `seq_len` directly), so -- unlike the standalone export's transformer, which
  requires the caller to pad captions to a multiple of 32 tokens -- `encode_prompt()` here does
  *not* pad.
- If multiple weight-precision variants exist for a component (e.g. `transformer_model_q4f16.onnx`
  vs. `transformer_model_q4f16_f32mix.onnx`), the q4f16 one is preferred; this is simply the
  literal filename hardcoded below, so the other variant is never even considered.

Pipeline flow -- one box per ONNX model under `<model>/onnx/`, run in this order once per
image (the transformer/scheduler_step pair loops `num_inference_steps` times):

    prompt
      |
      v
    HF tokenizer (<model>/tokenizer/, not ONNX)
      | input_ids, attention_mask
      v
    +--------------------------------+
    | text_encoder_model_q4f16.onnx  |   input_ids, attention_mask -> encoder_hidden_state
    +--------------------------------+
      | encoder_hidden_states  (sliced to prompt length)
      v
    latents (random noise, [1,16,1,H/8,W/8]) ------------------------------+
      |                                                                    |
      |   +===================== per denoising step =====================+
      |   |                                                                |
      v   v                                                                |
    +--------------------------------+                                    |
    | transformer_model_q4f16.onnx   | <- timestep                        |
    +--------------------------------+                                    |
      | noise_pred ("unified_results", [16,1,H,W], no batch axis)         |
      v                                                                    |
    +--------------------------------+                                    |
    | scheduler_step_model_f16.onnx  | <- latents, step_info=[step, N]     |
    +--------------------------------+                                    |
      | latents (updated, [1,16,1,H,W])                                   |
      +---- next step (feeds back into transformer) ----------------------+
      |
      v  (after the final step)
    +--------------------------------+
    | vae_pre_process_model_f16.onnx |   latents -> scaled_latents (/scale + shift, frame axis dropped)
    +--------------------------------+
      | scaled_latents
      v
    +--------------------------------+
    | vae_decoder_model_f16.onnx     |   latent_sample -> sample ([1,3,H,W], range [-1,1])
    +--------------------------------+
      | sample
      v
    PNG file
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
import torch
from PIL import Image
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "z-image-turbo-webnn"

VAE_SCALING_FACTOR = 0.3611
VAE_SHIFT_FACTOR = 0.1159


def peak_memory_mb() -> float:
    mem_info = psutil.Process(os.getpid()).memory_info()
    # peak_wset is Windows-only; fall back to current RSS on Linux/macOS.
    peak_bytes = getattr(mem_info, "peak_wset", mem_info.rss)
    return peak_bytes / (1024 * 1024)


def input_dtype(session: ort.InferenceSession, name: str) -> np.dtype:
    onnx_type = next(i.type for i in session.get_inputs() if i.name == name)
    return np.float16 if onnx_type == "tensor(float16)" else np.float32


def select_webgpu_device(gpu: int) -> "ort.OrtEpDevice":
    # Multi-GPU machines can expose more than one WebGPU-capable adapter (e.g. an integrated +
    # a discrete GPU); --gpu picks which one by index into this list, mirroring the "GPU"
    # device-type link in the webnn-developer-preview demo (index.js's `?devicetype=gpu`),
    # which otherwise just takes whatever adapter `navigator.gpu.requestAdapter()` defaults to.
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "WebGpuExecutionProvider"]
    if not devices:
        raise RuntimeError("WebGPU requested but no WebGPU-capable device was found.")

    print("Available WebGPU devices:")
    for i, d in enumerate(devices):
        hw = d.device
        print(f"  [{i}] {hw.vendor}")
        print(f"      vendor_id: 0x{hw.vendor_id:04x}")
        print(f"      device_id: 0x{hw.device_id:04x}")
        for key, value in hw.metadata.items():
            print(f"      {key}: {value}")

    if not 0 <= gpu < len(devices):
        raise ValueError(f"--gpu {gpu} out of range (found {len(devices)} WebGPU device(s)).")

    chosen = devices[gpu]
    print(f"Using WebGPU device [{gpu}]: {chosen.device.vendor}")
    return chosen


def log_session_io(label: str, session: ort.InferenceSession) -> None:
    print(f"{label}")
    print("  input:")
    for i in session.get_inputs():
        print(f"    {i.name}: {i.type} {i.shape}")
    print("  output:")
    for o in session.get_outputs():
        print(f"    {o.name}: {o.type} {o.shape}")


def to_uint8_hwc(vae_output: np.ndarray) -> np.ndarray:
    # vae_output: (1, 3, H, W) float, normalized to [-1, 1].
    chw = vae_output[0].astype(np.float32)
    chw = np.clip(chw * 0.5 + 0.5, 0.0, 1.0)
    return (chw * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0)


def save_image(hwc: np.ndarray, path: str) -> None:
    Image.fromarray(hwc, mode="RGB").save(path)
    print(f"Image saved to {path} ({os.path.getsize(path) / 1024:.1f} KB)")


class Scheduler:
    """Flow-matching (shift=3) timestep schedule -- must match the sigma schedule baked into
    scheduler_step_model_f16.onnx (see build_helper_models.py)."""

    NUM_TRAIN_TIMESTEPS = 1000
    SHIFT = 3.0

    def timesteps(self, num_inference_steps: int) -> np.ndarray:
        sigma = np.linspace(1.0, self.NUM_TRAIN_TIMESTEPS, self.NUM_TRAIN_TIMESTEPS, dtype=np.float32)[::-1] / self.NUM_TRAIN_TIMESTEPS
        sigma_max, sigma_min = sigma[0], sigma[-1]

        t = np.linspace(sigma_max * self.NUM_TRAIN_TIMESTEPS, sigma_min * self.NUM_TRAIN_TIMESTEPS, num_inference_steps)
        sigmas = t / self.NUM_TRAIN_TIMESTEPS
        sigmas = self.SHIFT * sigmas / (1 + (self.SHIFT - 1) * sigmas)

        # Transformer expects timestep in [0, 1] (0 = noise, 1 = clean), the reverse convention
        # of the sigma schedule above.
        timesteps = (self.NUM_TRAIN_TIMESTEPS - sigmas * self.NUM_TRAIN_TIMESTEPS) / self.NUM_TRAIN_TIMESTEPS
        timesteps[-1] = 1.0
        return timesteps.astype(np.float32)


class ZImagePipeline:
    def __init__(self, model_dir: str, ep: str, gpu: int = 0, sync: bool = False):
        self.sync = sync
        available = ort.get_available_providers()
        use_webgpu = ep == "WebGPU" or (not ep and "WebGpuExecutionProvider" in available)
        if ep == "WebGPU" and "WebGpuExecutionProvider" not in available:
            raise RuntimeError("WebGPU requested but not available in this onnxruntime build.")

        if use_webgpu:
            print("Execution provider: WebGpuExecutionProvider")
            webgpu_device = select_webgpu_device(gpu)
            sess_options = ort.SessionOptions()
            sess_options.add_provider_for_devices([webgpu_device], {})
            session_kwargs = {"sess_options": sess_options}
            # Device to bind per-step outputs (latents/noise_pred) to via IO binding, so they
            # stay GPU-resident across the denoising loop instead of round-tripping to host
            # memory between every model call -- mirrors the webnn-developer-preview demo's
            # `useIOBinding` GPU-buffer tensors (index.js's createGpuTensor/gpuBuffer) and this
            # repo's ort_playground/py/onnxruntime/llm/llm-ort.py KV-cache binding.
            self.device_type, self.device_id = "webgpu", gpu
        else:
            print("Execution provider: CPUExecutionProvider")
            session_kwargs = {"providers": ["CPUExecutionProvider"]}
            self.device_type, self.device_id = "cpu", 0

        onnx_dir = os.path.join(model_dir, "onnx")
        self.text_encoder = ort.InferenceSession(os.path.join(onnx_dir, "text_encoder_model_q4f16.onnx"), **session_kwargs)
        self.transformer = ort.InferenceSession(os.path.join(onnx_dir, "transformer_model_q4f16.onnx"), **session_kwargs)
        self.scheduler_step = ort.InferenceSession(os.path.join(onnx_dir, "scheduler_step_model_f16.onnx"), **session_kwargs)
        self.vae_pre_process = ort.InferenceSession(os.path.join(onnx_dir, "vae_pre_process_model_f16.onnx"), **session_kwargs)
        self.vae_decoder = ort.InferenceSession(os.path.join(onnx_dir, "vae_decoder_model_f16.onnx"), **session_kwargs)

        self.text_encoder_iob = self.text_encoder.io_binding()
        self.transformer_iob = self.transformer.io_binding()
        self.scheduler_step_iob = self.scheduler_step.io_binding()
        self.vae_pre_process_iob = self.vae_pre_process.io_binding()
        self.vae_decoder_iob = self.vae_decoder.io_binding()

        print("Model shapes:")
        log_session_io("text_encoder", self.text_encoder)
        log_session_io("transformer", self.transformer)
        log_session_io("scheduler_step", self.scheduler_step)
        log_session_io("vae_pre_process", self.vae_pre_process)
        log_session_io("vae_decoder", self.vae_decoder)

        self.transformer_dtype = input_dtype(self.transformer, "hidden_states")
        self.scheduler_step_dtype = input_dtype(self.scheduler_step, "latents")
        self.vae_pre_process_dtype = input_dtype(self.vae_pre_process, "latents")
        self.vae_decoder_dtype = input_dtype(self.vae_decoder, "latent_sample")

        self.text_encoder_output = self.text_encoder.get_outputs()[0].name
        self.transformer_output = self.transformer.get_outputs()[0].name
        self.scheduler_step_output = self.scheduler_step.get_outputs()[0].name
        self.vae_pre_process_output = self.vae_pre_process.get_outputs()[0].name
        self.vae_decoder_output = self.vae_decoder.get_outputs()[0].name

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.scheduler = Scheduler()

    def to_ort_value(self, array: np.ndarray) -> ort.OrtValue:
        # Host-side numpy buffers can only back a CPU OrtValue -- the pip onnxruntime packages
        # don't support allocating device memory directly from a numpy array for GPU EPs like
        # WebGPU (see llm-ort.py's module docstring). IO binding's bind_output(device_type=...)
        # is how the device-side buffer actually gets created instead.
        return ort.OrtValue.ortvalue_from_numpy(array)

    def _run_bound(
        self, label: str, session: ort.InferenceSession, iob: "ort.IOBinding", output_names: list, inputs: dict
    ) -> list:
        # Mirrors ort_playground's llm-ort.py: numpy inputs are bound host-side (bind_cpu_input,
        # ORT copies them to the device itself), OrtValue inputs are already wherever a previous
        # bind_output landed (bind_ortvalue_input, no copy), and every output is bound with
        # bind_output(device_type=...) so ORT allocates it directly on the target device instead
        # of the default host allocation run_with_ort_values would use.
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                iob.bind_cpu_input(name, value)
            else:
                iob.bind_ortvalue_input(name, value)
        for name in output_names:
            iob.bind_output(name, device_type=self.device_type, device_id=self.device_id)
        if self.sync:
            iob.synchronize_inputs()
        session.run_with_iobinding(iob)
        outputs = iob.get_outputs()
        if self.sync:
            # WebGPU dispatch is asynchronous: run_with_iobinding only submits the compute to the
            # GPU queue and returns immediately. IOBinding.synchronize_outputs() alone doesn't
            # force a wait for outputs that stay device-resident -- reading each one back to
            # host does, so that's the only reliable way to get a per-call time that reflects
            # actual compute instead of just CPU-side submission overhead. Off by default since
            # it serializes every step (each call blocks on the previous step's GPU work).
            iob.synchronize_outputs()
            for value in outputs:
                value.numpy()
        self._log_io(label, inputs, output_names, outputs)
        return outputs

    def _device_label(self, value: ort.OrtValue) -> str:
        # OrtValue.device_name() only ever reports "cpu" or "cuda" -- "cuda" is a legacy label
        # onnxruntime's python bindings use for any non-CPU OrtDevice (WebGPU/DML/ROCm/CUDA all
        # share the same generic GPU device-type enum), not an indication it ran on CUDA. Relabel
        # it using the EP we actually configured the pipeline with.
        name = value.device_name()
        return self.device_type if name == "cuda" else name

    def _log_io(self, label: str, inputs: dict, output_names: list, outputs: list) -> None:
        print(f"  [{label}] io:")
        for name, value in inputs.items():
            device = self._device_label(value) if isinstance(value, ort.OrtValue) else "cpu"
            print(f"    in  {name}: {device}")
        for name, value in zip(output_names, outputs):
            print(f"    out {name}: {self._device_label(value)}")

    def encode_prompt(self, prompt: str) -> ort.OrtValue:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="np")
        input_ids = inputs.input_ids.astype(np.int64)
        attention_mask = inputs.attention_mask.astype(np.int64)

        seq_len = int(attention_mask.sum())
        print(f"Tokenized prompt: {seq_len} tokens")

        embeds_ov = self._run_bound(
            "text_encoder", self.text_encoder, self.text_encoder_iob, [self.text_encoder_output],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]

        # This bundle's transformer has its own internal attention-mask/padding logic (no
        # separate mask input, but it accepts any seq_len directly), so -- unlike the standalone
        # export's transformer -- no padding to a multiple of 32 tokens is needed here.
        embeds = embeds_ov.numpy()[:, :seq_len, :]
        embeds_ov = self.to_ort_value(embeds.astype(self.transformer_dtype))
        print(f"prompt_embeds device: {embeds_ov.device_name()}")
        return embeds_ov

    def run(
        self,
        prompt: str,
        output_path: str,
        num_inference_steps: int,
        height: int,
        width: int,
        all_images: bool = False,
        seed: int = 42,
    ) -> None:
        print(f"height: {height}, width: {width}, steps: {num_inference_steps}")

        latent_h, latent_w = height // 8, width // 8
        # This bundle's latents carry an explicit num_frames=1 axis: [1, 16, 1, H, W] (rank 5).
        latents = torch.randn(
            (1, 16, 1, latent_h, latent_w), generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32
        ).numpy()
        latents_ov = self.to_ort_value(latents.astype(self.transformer_dtype))
        print(f"latents device: {latents_ov.device_name()}")
        timesteps = self.scheduler.timesteps(num_inference_steps)

        total_ms = 0.0

        def timed(label, fn, *a):
            nonlocal total_ms
            start = time.perf_counter()
            result = fn(*a)
            ms = (time.perf_counter() - start) * 1000
            if self.sync:
                # Without --sync these per-call numbers are meaningless (WebGPU dispatch is
                # async -- see _run_bound), so only show them when they're actually trustworthy.
                print(f"{label} time: {ms:.2f} ms")
            total_ms += ms
            return result

        print(f"Prompt:\n{prompt}")
        prompt_embeds = timed("text_encoder", self.encode_prompt, prompt)

        for step in range(num_inference_steps):
            timestep = timesteps[step]

            noise_pred_ov = timed(f"transformer-{step}", self._run_transformer, latents_ov, timestep, prompt_embeds)
            latents_ov = timed(f"scheduler_step-{step}", self._run_scheduler_step, noise_pred_ov, latents_ov, step, num_inference_steps)

            if all_images and step < num_inference_steps - 1:
                path = Path(output_path)
                hwc = to_uint8_hwc(self._decode(latents_ov).numpy())
                save_image(hwc, str(path.with_name(f"{path.stem}-step{step}{path.suffix}")))

        scaled_latents_ov = timed("vae_pre_process", self._run_vae_pre_process, latents_ov)
        image_ov = timed("vae_decoder", self._run_vae_decoder, scaled_latents_ov)

        def _postprocess():
            # image_ov.numpy() is the read that forces the final GPU sync -- keep it (and the
            # uint8/HWC conversion) inside the timed window so total_ms always reflects the full
            # pipeline, not just whichever step happens to force a sync first (see CHANGELOG.md).
            # The actual PNG encode + disk write below is excluded -- that's file I/O, not part
            # of the inference pipeline.
            return to_uint8_hwc(image_ov.numpy())

        hwc = timed("postprocess", _postprocess)
        save_image(hwc, output_path)
        print(f"total time: {total_ms:.2f} ms")

    def _run_transformer(self, latents_ov: ort.OrtValue, timestep: float, prompt_embeds_ov: ort.OrtValue) -> ort.OrtValue:
        inputs = {
            "hidden_states": latents_ov,
            "timestep": np.array([timestep], dtype=self.transformer_dtype),
            "encoder_hidden_states": prompt_embeds_ov,
        }
        return self._run_bound("transformer", self.transformer, self.transformer_iob, [self.transformer_output], inputs)[0]

    def _run_scheduler_step(
        self, noise_pred_ov: ort.OrtValue, latents_ov: ort.OrtValue, step: int, num_inference_steps: int
    ) -> ort.OrtValue:
        inputs = {
            "noise_pred": noise_pred_ov,
            "latents": latents_ov,
            "step_info": np.array([step, num_inference_steps], dtype=self.scheduler_step_dtype),
        }
        return self._run_bound(
            "scheduler_step", self.scheduler_step, self.scheduler_step_iob, [self.scheduler_step_output], inputs
        )[0]

    def _run_vae_pre_process(self, latents_ov: ort.OrtValue) -> ort.OrtValue:
        return self._run_bound(
            "vae_pre_process", self.vae_pre_process, self.vae_pre_process_iob, [self.vae_pre_process_output],
            {"latents": latents_ov},
        )[0]

    def _run_vae_decoder(self, scaled_latents_ov: ort.OrtValue) -> ort.OrtValue:
        if self.vae_pre_process_dtype != self.vae_decoder_dtype:
            # One-off dtype mismatch fixup (e.g. a float16 vae_pre_process feeding a float32
            # vae_decoder); not part of the per-step loop, so a host round-trip is fine here.
            scaled_latents_ov = self.to_ort_value(scaled_latents_ov.numpy().astype(self.vae_decoder_dtype))
        return self._run_bound(
            "vae_decoder", self.vae_decoder, self.vae_decoder_iob, [self.vae_decoder_output],
            {"latent_sample": scaled_latents_ov},
        )[0]

    def _decode(self, latents_ov: ort.OrtValue) -> ort.OrtValue:
        return self._run_vae_decoder(self._run_vae_pre_process(latents_ov))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Z-Image-Turbo inference against the legacy WebNN bundle in z-image-turbo-webnn/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model", nargs="?", default=str(DEFAULT_MODEL_DIR),
        help="Path to the legacy WebNN bundle directory (containing onnx/ and tokenizer/).",
    )
    parser.add_argument("--ep", default="", choices=["WebGPU", "CPU"], help="Execution provider.")
    parser.add_argument("--gpu", type=int, default=0, help="WebGPU device index, for machines with more than one GPU. Ignored with --ep CPU.")
    parser.add_argument(
        "--prompt",
        default="In a tranquil garden at dusk, a young Chinese woman stands gracefully in a red Hanfu with gold embroidery. Her flawless complexion features a red floral pattern on her forehead, enhancing her warm smile and expressive eyes. With her hair styled in a high bun adorned with a golden phoenix headdress, she holds a round folding fan decorated with nature scenes. Cherry blossom trees surround her, their petals drifting in the breeze, while a silhouetted pagoda (西安大雁塔) adds depth, blending tradition with modernity.",
        help="Text prompt to generate the image from.",
    )
    parser.add_argument("-s", "--step", type=int, default=4, help="Number of denoising steps.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("-o", "--output_name", default="z-image-turbo.png", help="Output image path.")
    parser.add_argument("-l", "--loop", type=int, default=1, help="Number of times to repeat generation (for benchmarking).")
    parser.add_argument("-a", "--all_images", action="store_true", help="Also write an image after every denoising step.")
    parser.add_argument(
        "--sync", action="store_true",
        help="Force a GPU sync after every model call for accurate per-call timing (default: off; "
        "WebGPU dispatch is otherwise async, so per-call times would only reflect submission "
        "overhead, not real compute time -- see CHANGELOG.md).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Latent noise seed.")
    return parser.parse_args()


def main():
    # Windows consoles default stdout to the system codepage (e.g. cp1252), which can't encode
    # prompts containing non-Latin-1 characters (the default prompt has Chinese in it).
    sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(f"Model path not found: {args.model}")
    if not os.path.isdir(os.path.join(args.model, "tokenizer")):
        raise SystemExit(f"Tokenizer not found under {args.model}/tokenizer.")

    pipeline = ZImagePipeline(args.model, args.ep, args.gpu, args.sync)

    output_name = Path(args.output_name)
    stem = f"{output_name.stem}_{args.width}x{args.height}_steps{args.step}"
    for i in range(args.loop):
        loop_name = output_name.with_name(f"{stem}_loop{i}{output_name.suffix}")
        pipeline.run(args.prompt, str(loop_name), args.step, args.height, args.width, args.all_images, args.seed)

    print(f"Peak Memory: {peak_memory_mb():.2f} MB")


if __name__ == "__main__":
    main()
