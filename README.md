# Z-Image-Turbo ONNX Exporters (onnxruntime-genai dev branch)

This branch of `onnxruntime-genai` adds standalone ONNX exporters for three pieces of the
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) text-to-image pipeline:

- the **diffusion transformer trunk** (diffusers class `ZImageTransformer2DModel`) — a custom
  exporter that consumes pre-computed caption embeddings and a raw image latent and produces a
  denoised/velocity-predicted latent;
- the **Qwen3 text encoder** that produces those caption embeddings — a custom exporter that
  builds a single-forward encoder graph directly from the Qwen3 checkpoint; and
- the **VAE decoder** (diffusers `AutoencoderKL.decoder`) that turns the final latent into an
  RGB image — see [ZIMAGE_VAE_USAGE.md](src/python/py/models/builders/ZIMAGE_VAE_USAGE.md).

None is an onnxruntime-genai runtime integration — all are plain ONNX graphs run directly
via `onnxruntime.InferenceSession`. A caller drives the diffusion sampling loop itself.

All of the exporter code lives under `src/python/py/models/`:

| File | Contents |
|---|---|
| [`src/python/py/models/builders/zimage.py`](src/python/py/models/builders/zimage.py) | `ZImageTransformerModel`, the transformer-trunk exporter itself |
| [`src/python/py/models/builders/zimage_text_encoder.py`](src/python/py/models/builders/zimage_text_encoder.py) | `export_qwen3_text_encoder`, builds the Qwen3 single-forward text encoder directly from the HF checkpoint |
| [`src/python/py/models/builders/zimage_vae.py`](src/python/py/models/builders/zimage_vae.py) | `ZImageVAEDecoderModel`, the VAE decoder exporter (see [ZIMAGE_VAE_DESIGN.md](src/python/py/models/builders/ZIMAGE_VAE_DESIGN.md) / [ZIMAGE_VAE_USAGE.md](src/python/py/models/builders/ZIMAGE_VAE_USAGE.md)) |
| [`src/python/py/models/build_z_image_turbo.py`](src/python/py/models/build_z_image_turbo.py) | CLI wrapper for building the transformer, text encoder, helper models, safety checker, VAE decoder, or all of them at once |
| [`src/python/py/models/builders/zimage_helper_models.py`](src/python/py/models/builders/zimage_helper_models.py) | `build_helper_models`, builds the scheduler_step/vae_pre_process/sc_prep helper graphs, shaped for the self-built transformer |
| [`src/python/py/models/builders/zimage_safety_checker.py`](src/python/py/models/builders/zimage_safety_checker.py) | `build_safety_checker`, exports the NSFW safety checker from the pretrained CLIP checkpoint |
| [`src/python/py/models/run_z_image_turbo.py`](src/python/py/models/run_z_image_turbo.py) | Standalone end-to-end text-to-image pipeline driver that can run the exported transformer |
| [`src/python/py/models/builders/ZIMAGE_DESIGN.md`](src/python/py/models/builders/ZIMAGE_DESIGN.md) | Architecture, scope, and design rationale |
| [`src/python/py/models/builders/ZIMAGE_USAGE.md`](src/python/py/models/builders/ZIMAGE_USAGE.md) | Full build/run/verify walkthrough, using `builder.py` directly |
| [`src/python/py/models/DESIGN.md`](src/python/py/models/DESIGN.md) | Design of the general model-builder pipeline this exporter reuses pieces of |

## Scope

- **Transformer trunk + Qwen3 text encoder + VAE decoder.** Each is built and swapped in
  independently; the VAE decoder is documented separately in `ZIMAGE_VAE_*.md`.
- **Standalone ONNX graphs.** No onnxruntime-genai C++ generator runtime integration.
- **Dynamic height/width**, batch size fixed at 1 (transformer).
- **No padding/pad-token machinery** in the transformer — resolutions and caption lengths must
  already satisfy the precondition below.

See [ZIMAGE_DESIGN.md#scope](src/python/py/models/builders/ZIMAGE_DESIGN.md#scope) for the
full list of what's out of scope (SigLIP/Omni conditioning, LoRA, ControlNet, multiple patch
sizes, gradient checkpointing).

## Quick Start

```bash
cd src/python/py/models
pip install diffusers pillow  # in addition to this repo's normal torch/onnx_ir/transformers/onnxruntime deps
```

Download the checkpoint:

```py
from huggingface_hub import snapshot_download
snapshot_download("Tongyi-MAI/Z-Image-Turbo", local_dir="path_to_local_folder")
```

This is enough to build the transformer, text encoder, helper models, and VAE decoder.
`-m safety_checker` and `-m all` also need a second, separate checkpoint — see
[Building the Safety Checker](#building-the-safety-checker) below.

Everything is built with [`build_z_image_turbo.py`](src/python/py/models/build_z_image_turbo.py),
which wraps `builder.py` with the WebGPU EP and the extra options each component needs
pre-filled in. `-m/--model` selects which component to build — `transformer` (default),
`text_encoder`, `helper_models`, `safety_checker`, `vae_decoder`, or `all` — see the matching
section below for each, or [Building Everything at Once](#building-everything-at-once) for the
one-command path.

## Building the Transformer

Pick one of four precisions with `-p`:

```bash
python build_z_image_turbo.py path_to_local_folder -m transformer -p <precision>
```

| `-p` value | `builder.py -p` | I/O dtype | Weights | Notes |
|---|---|---|---|---|
| `f16` | `fp16` | float16 | unquantized | |
| `f32` | `fp32` | float32 | unquantized | |
| `f16_int4_quant` (default) | `int4` | float16 | int4 (`MatMulNBits`) | `block_size=32 accuracy_level=4` |
| `f32_int4_quant` | `int4` | float32 | int4 (`MatMulNBits`) | same as above + `use_webgpu_fp32=true` |

Output goes to `<model_name>-transformer-genai-wgpu-<precision>/`. All four variants are
verified to produce correct (non-NaN) output — see
[ZIMAGE_DESIGN.md#float16-dynamic-range-overflow](src/python/py/models/builders/ZIMAGE_DESIGN.md#float16-dynamic-range-overflow)
for the float16 NaN issue this required fixing.

For finer control (different EP, `int8`, custom `--extra_options`, etc.), call `builder.py`
directly — see
[ZIMAGE_USAGE.md#building-the-onnx-model](src/python/py/models/builders/ZIMAGE_USAGE.md#building-the-onnx-model).

## Building the Text Encoder

The caption embeddings the transformer consumes come from Z-Image-Turbo's Qwen3 text encoder.
Build it with the same wrapper, `-m text_encoder`:

```bash
python build_z_image_turbo.py path_to_local_folder -m text_encoder -p <precision>
```

Unlike the transformer, this doesn't shell out to `builder.py` at all: `export_qwen3_text_encoder`
in [`builders/zimage_text_encoder.py`](src/python/py/models/builders/zimage_text_encoder.py)
constructs the encoder graph directly from the Qwen3 checkpoint (`config.json` + safetensors
weights only — it never touches the tokenizer) with `onnx.helper`, building only the first
`num_hidden_layers - 1` decoder layers plus one extra layer's input norm (the final layer, final
norm, and LM head are never built at all, not built-then-stripped):

- taps the residual stream entering the last decoder layer's input norm (equivalent to
  HuggingFace's `hidden_states[-2]`) as the output,
- exposes it as a float16 `encoder_hidden_state` of shape `[1, seq, 2560]`, and
- takes only `input_ids` and `attention_mask` as graph inputs — `position_ids` is computed
  internally from `input_ids`, and there's no KV cache (a graph-capture-style mask-reformatting
  subgraph derives `seqlens_k`/`total_seq_len` from `attention_mask` for `GroupQueryAttention`).

`-p` selects the precision: `f16` (unquantized float16 weights and I/O) or `f16_int4_quant`
(int4-quantized `MatMul`/`Gather` weights via `MatMulNBitsQuantizer`, float16 I/O). `f32` and
`f32_int4_quant` aren't supported (`GroupQueryAttention` under fp32 has never been verified for
this Qwen3 config) and fail fast with an error instead of silently building something else.

Output goes to `<model_name>-text_encoder-genai-wgpu-<precision>/text_encoder_model_<suffix>.onnx`
(+ `.onnx_data`), where `<suffix>` is `f16` or `q4f16` matching `-p`. The `q4f16` variant is a
drop-in for the WebNN bundle's `onnx/text_encoder_model_q4f16.onnx`.

> Q/K per-head RMSNorm stays as separate `SimplifiedLayerNormalization` ops rather than fused
> into `GroupQueryAttention`, so GQA keeps ≤12 inputs (rotary stays fused inside GQA). A fused
> form would emit a 16-input GQA that current onnxruntime-web / onnxruntime 1.24 reject at load.

## Building the Helper Models

The WebNN bundle also ships 3 tiny helper ONNX graphs that keep intermediate tensors off the
CPU: `scheduler_step_model_f16.onnx` (the flow-matching Euler latent update),
`vae_pre_process_model_f16.onnx` (squeeze + VAE scale/shift before decode), and
`sc_prep_model_f16.onnx` (resize + CLIP normalize for the safety checker). Those are shaped for
the WebNN transformer's `[*, 16, 1, H, W]` latents (a `num_frames` axis our self-built
transformer doesn't have) and always use a float32 I/O boundary.

`-m helper_models` builds our own versions of these 3 graphs (via
[`builders/zimage_helper_models.py`](src/python/py/models/builders/zimage_helper_models.py)),
shaped like the self-built transformer's `[1, 16, H, W]` (no frame axis), in a genuine float16
I/O boundary as well as float32:

```bash
cd src/python/py/models
pip install onnxconverter_common  # in addition to the deps above
python build_z_image_turbo.py path_to_local_folder -m helper_models -p f16
```

`-p` only differentiates `f16`/`f32` here (the `_int4_quant` half is ignored, and a bare
`-p f32`/`f32_int4_quant` builds the f32 variant) — run it twice for both, the same as building
both transformer precisions. Output goes to
`<model_name>-helper_models-genai-wgpu-<f16|f32>/<scheduler_step|vae_pre_process|sc_prep>_model_<precision>.onnx`;
the build verifies each graph against a pure-numpy reference immediately after building it. The
math (sigma schedule, VAE scale/shift, CLIP normalization constants) matches the deployed bundle
graphs exactly — only the shape (no frame axis) and float16 boundary are new.

## Building the Safety Checker

The NSFW safety checker (the bundle's `sc_prep` companion) is a real pretrained CLIP ViT-L/14
vision classifier — a cosine-distance threshold check against 17 "concept" and 3 "special care"
reference embeddings — not something authored from scratch. It's exported from the standard
public checkpoint `CompVis/stable-diffusion-safety-checker` via
[`builders/zimage_safety_checker.py`](src/python/py/models/builders/zimage_safety_checker.py),
using `diffusers`' own `forward_onnx` method (with the `images`-masking input/output trimmed off,
since only `clip_input -> has_nsfw_concepts` is needed). Download the checkpoint once, the same
way as the main Z-Image-Turbo checkpoint:

```py
from huggingface_hub import snapshot_download
snapshot_download("CompVis/stable-diffusion-safety-checker", local_dir="path_to_safety_checker_folder")
```

Then build:

```bash
python build_z_image_turbo.py path_to_local_folder -m safety_checker -p f16 \
  --safety_checker_checkpoint path_to_safety_checker_folder
```

`-p` only differentiates `f16`/`f32`, same as `-m helper_models`. Output goes to
`<model_name>-safety_checker-genai-wgpu-<f16|f32>/safety_checker_model_<precision>.onnx`. Verified
against the deployed bundle's `safety_checker_model_f16.onnx`: every named weight tensor
(`concept_embeds`, `special_care_embeds`, the CLIP vision encoder weights, etc.) matches
bit-for-bit (mod float16 rounding) — same checkpoint — and `has_nsfw_concepts` agrees across
randomized inputs run through both graphs in `onnxruntime`.

## Building the VAE Decoder

The VAE decoder itself is documented in full in
[ZIMAGE_VAE_DESIGN.md](src/python/py/models/builders/ZIMAGE_VAE_DESIGN.md) /
[ZIMAGE_VAE_USAGE.md](src/python/py/models/builders/ZIMAGE_VAE_USAGE.md), including the
`fuse_group_norm` GroupNorm flavour and int4/int8 mid-block quantization. `-m vae_decoder` is a
thin convenience wrapper around that same `builders/zimage_vae.py` exporter, for when you just
want an unquantized `f16`/`f32` decoder without the extra options:

```bash
python build_z_image_turbo.py path_to_local_folder -m vae_decoder -p f16
```

`-p` only differentiates `f16`/`f32`, same as `-m helper_models`/`-m safety_checker`. Output goes
to `<model_name>-vae_decoder-genai-wgpu-<f16|f32>/vae_decoder_model_<precision>.onnx` — a
self-contained file (weights inline, no `.onnx.data`), a drop-in for the WebNN bundle's
`onnx/vae_decoder_model_f16.onnx`.

## Building Everything at Once

`-m all` builds the transformer, text encoder, helper models, safety checker, and VAE decoder in
one command, into a single bundle-shaped directory (`--safety_checker_checkpoint` is required).
Since the text encoder only supports `f16`/`f16_int4_quant` (see
[Building the Text Encoder](#building-the-text-encoder)), `-m all` fails fast with an error for
`-p f32`/`f32_int4_quant` before building anything, rather than building a partial bundle:

```bash
python build_z_image_turbo.py path_to_local_folder -m all -p f16_int4_quant \
  --safety_checker_checkpoint path_to_safety_checker_folder
```

Output goes to `<model_name>-genai-wgpu-<precision>/`, laid out like the WebNN bundle itself:

```
<model_name>-genai-wgpu-<precision>/
  onnx/
    transformer_model_<f16|q4f16>.onnx(.data)
    text_encoder_model_<f16|q4f16>.onnx(_data)
    scheduler_step_model_<f16>.onnx
    vae_pre_process_model_<f16>.onnx
    sc_prep_model_<f16>.onnx
    safety_checker_model_<f16>.onnx
    vae_decoder_model_<f16>.onnx
  tokenizer/
    ...
```

The transformer and text encoder filename suffixes both match `-p` exactly (`f16`->`f16`/
`f16_int4_quant`->`q4f16`, mirroring the WebNN bundle's own `transformer_model_q4f16.onnx`/
`text_encoder_model_q4f16.onnx` convention). Everything else's precision is derived from `-p`'s
f16-vs-f32 half, which for `-m all` is always `f16` since only `f16`/`f16_int4_quant` are
accepted — the VAE decoder gets no int4/int8 quant here (unquantized, decomposed GroupNorm,
matching the WebNN bundle's own `vae_decoder_model_f16.onnx`); for quantization or
`fuse_group_norm=true`, build it standalone with `-m vae_decoder` or call `builder.py` directly
(see [ZIMAGE_VAE_USAGE.md](src/python/py/models/builders/ZIMAGE_VAE_USAGE.md)).

This bundle is fully self-contained — no WebNN bundle needed at all. Point `run_z_image_turbo.py`
at it directly (see [Running the Full Pipeline](#running-the-full-pipeline-text-to-image)); for
the default `f16_int4_quant` precision, every filename above already matches the pipeline's
built-in defaults, so its swap-in flags (`--transformer`, `--text_encoder`, `--vae_decoder`,
`--scheduler_step`, `--vae_pre_process`, `--sc_prep`) are optional.

## Precondition on Resolution and Caption Length

Because the exported graph has no padding/masking logic, the caller must ensure:

- `(height / patch_size) * (width / patch_size) % 32 == 0` (`patch_size = 2`) — holds for
  every standard resolution divisible by 16 (512, 768, 1024, ...), including non-square
  combinations.
- The caption embedding's sequence length is already a multiple of 32 tokens.

Violating either precondition does not error at export time — it silently computes the
wrong thing at inference time. See
[ZIMAGE_USAGE.md#precondition-on-resolution-and-caption-length](src/python/py/models/builders/ZIMAGE_USAGE.md#precondition-on-resolution-and-caption-length).

## Running the Exported Graph

```py
import onnxruntime as ort

sess = ort.InferenceSession("path_to_output_folder/model.onnx", providers=["CPUExecutionProvider"])
sample = sess.run(
    ["sample"],
    {
        "hidden_states": hidden_states_np,          # [1, 16, H, W]
        "encoder_hidden_states": encoder_hidden_states_np,  # [1, cap_len, 2560]
        "timestep": timestep_np,                    # [1]
    },
)[0]
```

Full details, including a verification script that cross-checks the ONNX output against a
hand-written PyTorch reference, are in
[ZIMAGE_USAGE.md#running-the-exported-graph](src/python/py/models/builders/ZIMAGE_USAGE.md#running-the-exported-graph)
and
[ZIMAGE_USAGE.md#verifying-numerical-correctness](src/python/py/models/builders/ZIMAGE_USAGE.md#verifying-numerical-correctness).

## Running the Full Pipeline (Text-to-Image)

[`run_z_image_turbo.py`](src/python/py/models/run_z_image_turbo.py) is a standalone,
self-contained script that drives an actual end-to-end Z-Image-Turbo text-to-image
generation (tokenizer -> text encoder -> flow-matching denoising loop -> VAE decode -> PNG),
useful for exercising the exported transformer against real prompts instead of the synthetic
tensors used for verification.

It expects a WebNN-exported Z-Image-Turbo model directory (with `tokenizer/`,
`onnx/text_encoder_model_q4f16.onnx`, and `onnx/vae_decoder_model_f16.onnx`) for the
tokenizer/text-encoder/VAE baseline. Each of the three models can be swapped for a self-built
one with `--transformer`, `--text_encoder` and `--vae_decoder` (all are drop-ins — no need to
copy files over the bundle). A fully self-built [`-m all`](#building-everything-at-once) bundle
is a complete drop-in for this positional argument too, since it has the same `tokenizer/` +
`onnx/` layout:

```bash
cd src/python/py/models
pip install psutil transformers pillow torch onnxruntime  # in addition to the deps above

python run_z_image_turbo.py path_to_webnn_z_image_turbo_dir \
  --transformer path_to_transformer_output_folder/model.onnx \
  --text_encoder path_to_text_encoder_output_folder/text_encoder_model_q4f16.onnx \
  --vae_decoder path_to_vae_output_folder/model.onnx \
  --prompt "a cat under the snow with blue eyes, cinematic style" \
  --height 512 --width 512 \
  -n 4 -o output.png
```

The pipeline auto-detects each of these three models' shape convention and I/O dtype directly
from the loaded graph, not from whether the flag was passed — so it works whether a self-built
model was passed explicitly, or happens to already sit at the bundle's default filename (as in
an `-m all` bundle). `--transformer` (or a self-built `transformer_model_q4f16.onnx` sitting at
the default path) is detected as 4D `hidden_states` (no `num_frames` axis, unlike the bundled
WebNN transformer's 5D) and no attention mask/padding (`encoder_hidden_states` is padded to a
multiple of 32 tokens by repeating the last real token's embedding). `--text_encoder` swaps in a
`build_z_image_turbo.py -m text_encoder` encoder in place of the bundle's
`onnx/text_encoder_model_q4f16.onnx`; it's a drop-in (same `input_ids`/`attention_mask` inputs,
single float16 `encoder_hidden_state` output, auto-detected at load). `--vae_decoder` swaps in a
`builders/zimage_vae.py` export (same `latent_sample` -> `sample` interface as the bundle's
`onnx/vae_decoder_model_f16.onnx`; its float16/float32 I/O dtype is read from the model). Every
flag is optional — omit all to run the WebNN bundle end-to-end as a baseline, or pass only one to
isolate a single self-built component. See `--help` for `--ep` (WebGPU/CPU), `--all_images` (dump
every denoising step), `-l/--loop` (repeat generation), and `-v` (verbose per-tensor stats)
options.

To match what the WebNN browser demo actually runs, the flow-matching Euler step and the VAE
pre-scale are executed as the bundle's small helper graphs (`onnx/scheduler_step_model_f16.onnx`
and `onnx/vae_pre_process_model_f16.onnx`) rather than in numpy. `--safety_checker` additionally
runs the bundle's `onnx/sc_prep_model_f16.onnx` + `onnx/safety_checker_model_f16.onnx` after
decoding and prints an NSFW verdict; it loads ~580 MB extra and its runtime is reported
separately and excluded from the pipeline's total-time metric.

`--scheduler_step`, `--vae_pre_process`, and `--sc_prep` swap in self-built versions of those
first two helper graphs (see [Building the Helper Models](#building-the-helper-models)) the same
way `--transformer`/`--text_encoder` do — each is a drop-in; the pipeline auto-detects the
loaded graph's dtype and shape convention (frame-axis or not) at init time, so any mix of
bundle/self-built helpers works.

## Further Reading

- [ZIMAGE_DESIGN.md](src/python/py/models/builders/ZIMAGE_DESIGN.md) — why this doesn't fit
  the standard `Model` pipeline, RoPE/patchify/AdaLN graph construction, quantization
  coverage, the float16 overflow fix, and a symbolic-dim-aliasing implementation gotcha.
- [ZIMAGE_USAGE.md](src/python/py/models/builders/ZIMAGE_USAGE.md) — build/run/verify
  walkthrough and troubleshooting.
- [DESIGN.md](src/python/py/models/DESIGN.md) — the general model-builder pipeline
  (`Model`, `make_matmul`, quantization pass) this exporter reuses low-level pieces of.
