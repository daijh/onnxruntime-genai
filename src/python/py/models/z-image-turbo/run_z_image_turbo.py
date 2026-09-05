"""Reference driver for the standalone Z-Image-Turbo ONNX export in `z-image-turbo-onnx/`
(see export_models.py / README.md).

Unlike the older `../run_z_image_turbo.py` (written against the WebNN demo's bundled models,
which mix WebNN-shaped [B,16,1,H,W] 5-frame-axis tensors with dev-model [1,16,H,W] ones and
support swapping either in), every model here comes from this repo's own exporters and shares
one fixed shape convention: no frame axis, float16 I/O throughout except the tokenizer/text
encoder boundary. That removes the need for any shape/dtype auto-detection or path-swapping
flags.

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
      | encoder_hidden_states  (sliced to prompt length, padded to a multiple of 32 tokens)
      v
    latents (random noise, [1,16,H/8,W/8]) --------------------------------+
      |                                                                    |
      |   +===================== per denoising step =====================+
      |   |                                                                |
      v   v                                                                |
    +--------------------------------+                                    |
    | transformer_model_q4f16.onnx   | <- timestep                        |
    +--------------------------------+                                    |
      | noise_pred                                                        |
      v                                                                    |
    +--------------------------------+                                    |
    | scheduler_step_model_f16.onnx  | <- latents, step_info=[step, N]     |
    +--------------------------------+                                    |
      | latents (updated)                                                 |
      +---- next step (feeds back into transformer) ----------------------+
      |
      v  (after the final step)
    +--------------------------------+
    | vae_pre_process_model_f16.onnx |   latents -> scaled_latents (/scale + shift)
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
import resource
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "z-image-turbo-onnx"

VAE_SCALING_FACTOR = 0.3611
VAE_SHIFT_FACTOR = 0.1159


def peak_memory_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS.
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 if os.uname().sysname == "Linux" else kb / (1024 * 1024)


def input_dtype(session: ort.InferenceSession, name: str) -> np.dtype:
    onnx_type = next(i.type for i in session.get_inputs() if i.name == name)
    return np.float16 if onnx_type == "tensor(float16)" else np.float32


def log_session_io(label: str, session: ort.InferenceSession) -> None:
    print(f"{label}")
    print("  input:")
    for i in session.get_inputs():
        print(f"    {i.name}: {i.type} {i.shape}")
    print("  output:")
    for o in session.get_outputs():
        print(f"    {o.name}: {o.type} {o.shape}")


def save_image(vae_output: np.ndarray, path: str) -> None:
    # vae_output: (1, 3, H, W) float, normalized to [-1, 1].
    chw = vae_output[0].astype(np.float32)
    chw = np.clip(chw * 0.5 + 0.5, 0.0, 1.0)
    hwc = (chw * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0)
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
    def __init__(self, model_dir: str, ep: str, verbose: bool = False):
        self.verbose = verbose

        available = ort.get_available_providers()
        if not ep:
            providers = ["WebGpuExecutionProvider"] if "WebGpuExecutionProvider" in available else ["CPUExecutionProvider"]
        elif ep == "WebGPU":
            if "WebGpuExecutionProvider" not in available:
                raise RuntimeError("WebGPU requested but not available in this onnxruntime build.")
            providers = ["WebGpuExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        print(f"Execution provider: {providers[0]}")

        onnx_dir = os.path.join(model_dir, "onnx")
        self.text_encoder = ort.InferenceSession(os.path.join(onnx_dir, "text_encoder_model_q4f16.onnx"), providers=providers)
        self.transformer = ort.InferenceSession(os.path.join(onnx_dir, "transformer_model_q4f16.onnx"), providers=providers)
        self.scheduler_step = ort.InferenceSession(os.path.join(onnx_dir, "scheduler_step_model_f16.onnx"), providers=providers)
        self.vae_pre_process = ort.InferenceSession(os.path.join(onnx_dir, "vae_pre_process_model_f16.onnx"), providers=providers)
        self.vae_decoder = ort.InferenceSession(os.path.join(onnx_dir, "vae_decoder_model_f16.onnx"), providers=providers)

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

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.scheduler = Scheduler()

    def encode_prompt(self, prompt: str) -> np.ndarray:
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

        embeds = self.text_encoder.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
        embeds = embeds[:, :seq_len, :]

        # The transformer has no attention-mask/padding logic (see build_transformer.py), so
        # the caption length must already be a multiple of 32 tokens; pad by repeating the
        # last real token's embedding.
        pad_len = (-embeds.shape[1]) % 32
        if pad_len:
            pad = np.repeat(embeds[:, -1:, :], pad_len, axis=1)
            embeds = np.concatenate([embeds, pad], axis=1)

        return embeds.astype(self.transformer_dtype)

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
        latents = torch.randn(
            (1, 16, latent_h, latent_w), generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32
        ).numpy()
        timesteps = self.scheduler.timesteps(num_inference_steps)

        total_ms = 0.0

        def timed(label, fn, *a):
            nonlocal total_ms
            start = time.perf_counter()
            result = fn(*a)
            ms = (time.perf_counter() - start) * 1000
            print(f"{label} time: {ms:.2f} ms")
            total_ms += ms
            return result

        print(f"Prompt:\n{prompt}")
        prompt_embeds = timed("text_encoder", self.encode_prompt, prompt)

        for step in range(num_inference_steps):
            timestep = timesteps[step]

            noise_pred = timed(f"transformer-{step}", self._run_transformer, latents, timestep, prompt_embeds)
            latents = timed(f"scheduler_step-{step}", self._run_scheduler_step, noise_pred, latents, step, num_inference_steps)

            if all_images and step < num_inference_steps - 1:
                path = Path(output_path)
                save_image(self._decode(latents), str(path.with_name(f"{path.stem}-step{step}{path.suffix}")))

        scaled_latents = timed("vae_pre_process", self._run_vae_pre_process, latents)
        image = timed("vae_decoder", self._run_vae_decoder, scaled_latents)
        save_image(image, output_path)
        print(f"total time: {total_ms:.2f} ms")

    def _run_transformer(self, latents: np.ndarray, timestep: float, prompt_embeds: np.ndarray) -> np.ndarray:
        ort_inputs = {
            "hidden_states": latents.astype(self.transformer_dtype),
            "timestep": np.array([timestep], dtype=self.transformer_dtype),
            "encoder_hidden_states": prompt_embeds,
        }
        return self.transformer.run(None, ort_inputs)[0]

    def _run_scheduler_step(self, noise_pred: np.ndarray, latents: np.ndarray, step: int, num_inference_steps: int) -> np.ndarray:
        dtype = self.scheduler_step_dtype
        ort_inputs = {
            "noise_pred": noise_pred.astype(dtype),
            "latents": latents.astype(dtype),
            "step_info": np.array([step, num_inference_steps], dtype=dtype),
        }
        return self.scheduler_step.run(None, ort_inputs)[0].astype(np.float32)

    def _run_vae_pre_process(self, latents: np.ndarray) -> np.ndarray:
        return self.vae_pre_process.run(None, {"latents": latents.astype(self.vae_pre_process_dtype)})[0]

    def _run_vae_decoder(self, scaled_latents: np.ndarray) -> np.ndarray:
        return self.vae_decoder.run(None, {"latent_sample": scaled_latents.astype(self.vae_decoder_dtype)})[0]

    def _decode(self, latents: np.ndarray) -> np.ndarray:
        return self._run_vae_decoder(self._run_vae_pre_process(latents))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Z-Image-Turbo inference against the standalone ONNX export in z-image-turbo-onnx/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model", nargs="?", default=str(DEFAULT_MODEL_DIR),
        help="Path to the exported ONNX model directory (containing onnx/ and tokenizer/, as written by export_models.py).",
    )
    parser.add_argument("--ep", default="", choices=["WebGPU", "CPU"], help="Execution provider.")
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--seed", type=int, default=42, help="Latent noise seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(f"Model path not found: {args.model}")
    if not os.path.isdir(os.path.join(args.model, "tokenizer")):
        raise SystemExit(
            f"Tokenizer not found under {args.model}/tokenizer -- re-run export_models.py to "
            "populate it."
        )

    pipeline = ZImagePipeline(args.model, args.ep, args.verbose)

    output_name = Path(args.output_name)
    stem = f"{output_name.stem}_{args.width}x{args.height}_steps{args.step}"
    for i in range(args.loop):
        loop_name = output_name.with_name(f"{stem}_loop{i}{output_name.suffix}")
        pipeline.run(args.prompt, str(loop_name), args.step, args.height, args.width, args.all_images, args.seed)

    print(f"Peak Memory: {peak_memory_mb():.2f} MB")


if __name__ == "__main__":
    main()
