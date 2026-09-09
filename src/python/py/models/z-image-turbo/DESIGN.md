# Z-Image-Turbo Standalone Exporters — Design Notes

Rationale, non-obvious tradeoffs, and workarounds for the code in this folder. See
`README.md` for usage; this doc explains *why* the code is shaped the way it is.

## Overview

Every `build_*.py` here is a **self-contained port** of the corresponding module in
`../builders/` (`zimage.py`, `zimage_text_encoder.py`, `zimage_vae.py`,
`zimage_helper_models.py`, `zimage_safety_checker.py`). Unlike those, none of these depend on
this repo's own `../builders/base.py` source tree — each depends only on the *public*
`onnxruntime-genai` pip package (plus `torch`/`onnx`/`diffusers`/`transformers` as needed). That
means each `build_*.py` file can be handed out on its own — `pip install -r requirements.txt`
and run it — with no checkout of this repo required. This is why some boilerplate (e.g.
`set_io_dtype`/`set_onnx_dtype`/`load_diffusers_config` in `build_transformer.py` and
`build_vae_decoder.py`, or `parse_extra_options` across several files) is duplicated
verbatim rather than factored into a shared module: a shared module would break that
single-file portability.

`export_models.py` is the dispatcher: `-m/--model` selects which `build_*.py`'s `build()` to
call (`transformer`/`text_encoder`/`vae_decoder`/`helper_models`/`safety_checker`, or `all` for
every component into one bundle directory). `-m all` runs each component in its own subprocess
(`build_component_subprocess`) so one component's import-time state/crashes can't affect
another. `safety_checker` needs a separate checkpoint download (`--safety_checker_checkpoint`,
a `CompVis/stable-diffusion-safety-checker`-compatible folder) since it isn't part of the
Z-Image-Turbo checkpoint itself; `-m all` just skips it if that flag isn't given.

The Z-Image-Turbo checkpoint's tokenizer lives in a `tokenizer/` folder that's a *sibling* of
`text_encoder/`, not inside it, regardless of whether the path handed to `export_models.py` is
the checkpoint repo root or a component subfolder directly (both are auto-detected —
`resolve_tokenizer_dir`). `copy_tokenizer` copies those files into `<output_dir>/tokenizer/` so
the exported directory is self-contained and can be pointed at directly by
`run_z_image_turbo.py`.

Batch size is fixed at 1 throughout the whole pipeline (every graph input/output that has a
batch dim declares it as a static `1`, not a symbolic dim) — this is a deliberate scope
decision (see README.md), not an oversight, and it's what lets ONNX Runtime's graph
optimizations fold more aggressively than a dynamic batch dim would allow.

## Transformer (`build_transformer.py`)

This model is a diffusion DiT, not a causal LM: it has three named transformer stacks
(`noise_refiner`, `context_refiner`, `layers`), bidirectional attention, AdaLN-style timestep
modulation, and 3-axis real-valued RoPE gathered from precomputed per-axis frequency tables.
None of that fits `Model.make_model`'s reflection-based causal-LM decoder loop, so
`ZImageTransformerModel` subclasses `Model` only to reuse its low-level ONNX node builders
(`make_matmul` for quantization, `make_multi_head_attention`, `save_model`/`ir.Model`
bookkeeping) — the top-level graph construction, weight loading, and I/O are all overridden
from scratch. No `genai_config.json` is produced: this is a standalone ONNX graph, not a genai
C++ runtime integration.

Scope: transformer trunk only (no text encoder / VAE), dynamic height/width, batch size fixed
at 1, and no padding/pad-token machinery — the caller must use resolutions and caption lengths
where token counts are already multiples of 32, which drops the need for
`x_pad_token`/`cap_pad_token`/attention masks.

**Fake config translation.** `Model.__init__` assumes a `transformers`-style causal-LM config
(`hidden_size`/`num_attention_heads`/`vocab_size`/`architectures`/...). The Z-Image-Turbo
diffusers config exposes none of that (it uses `dim`/`n_heads`/`_class_name`/etc., no
vocab/context-length concept), so `__init__` translates it into a minimal fake namespace that
satisfies every `hasattr`/attribute access in `Model.__init__` and its `make_*_init` helpers.
Everything LLM-specific that sets up (mask_attrs, rope_attrs, kv_cache_attrs, mlp_attrs,
moe_attrs, the standard input/output dicts, ...) is simply unused by this class.

**float16 overflow workaround** (`pre_out_proj_scale`/`ff_gate_scale`/`ff_up_scale`,
`_rescale_preout`). `attention.to_out`/`feed_forward.w2`'s raw (pre-
`SimplifiedLayerNormalization`) input can reach ~1e6 in magnitude on real inputs — this
overflows float16 (max ~65504) before the following RMSNorm ever gets a chance to renormalize
it back down, producing Inf -> NaN. Since `RMSNorm(x)` is exactly scale-invariant to a positive
scalar multiple of its input, rescaling *into* `to_out`/`w2` by a constant is a no-op on the
eventual normalized output (in exact math, and to well within float16 precision, since
`norm_eps` is negligible next to these signals' variance either way) while keeping every
intermediate representable in float16. All factors are exact powers of two in float16:
`attention.to_out` takes the full 1/128 on its single input (`_make_attention`).
`feed_forward.w2`'s input is the SwiGLU product `SiLU(w1) * w3`, so the same 1/128 is *split*
across the two factors — 1/8 on the SiLU gate, 1/16 on w3 (8 * 16 == 128) — and applied
*before* the elementwise multiply, so neither the product nor the w2 matmul that consumes it
can overflow (`_make_feed_forward`). No-op for float32 I/O, which has enough headroom that the
raw pre-norm magnitude never overflows — `_rescale_preout` skips the extra node there.

**`_layer_norm_no_affine`'s `stash_type=1`** (used for `FinalLayer.norm_final`) is not
cosmetic: it forces ONNX Runtime to compute the mean/variance — including the `(x-mean)^2`
accumulation — in float32 internally regardless of `io_dtype`. That range is required: the
pre-norm hidden state's squared deviation reaches ~2e6 on real inputs, which overflows float16
(max ~65504) to `Inf` and collapses the output to a garbled image on a true-float16 backend
(WebGPU/WebNN). The bug is masked on the CPU EP (which upcasts float16 math to float32) and
absent in the `-p f32*` builds, so it only surfaces on-device. `norm_final` has no learnable
affine, so `scale`/`bias` are materialized as all-ones/all-zeros (ONNX `LayerNormalization`
requires the scale input regardless); the AdaLN scale/shift is applied separately by the
caller.

**Node naming and `seq_dim`.** Node names follow a PyTorch-module-dotted-path convention
(`/model/layers.{i}/attention/to_q`, etc.) for anything mapping to a real submodule; purely
synthetic glue with no PyTorch counterpart (dynamic dims, RoPE table gathers, patchify/
unpatchify, position grids) lives under a separate `/model/z_image/...` namespace instead.
`_linear`'s `seq_dim` parameter matters for correctness, not just cosmetics: `make_matmul`/
`make_add_bias` always declare their output as the generic 3D `["batch_size", seq_dim,
"last_dim"]` shape, defaulting `seq_dim` to the shared literal `"sequence_length"`. This model
has three logically-distinct, differently-sized sequences in flight at once (image tokens,
caption tokens, unified tokens) plus several genuinely-2D tensors (AdaLN/timestep MLPs) —
reusing one shared symbolic name would make ONNX Runtime's memory planner treat same-named but
differently-sized intermediates as alias-compatible buffers, causing a runtime shape-mismatch
crash. So `_linear` derives a name unique per logical sequence (or, for 2D tensors, per call
site) unless the caller provides one explicitly, and re-stamps `make_matmul`'s declared shape
afterward since the generic 3D convention doesn't always match (2D AdaLN/timestep tensors);
`make_multi_head_attention`'s output needs the same re-stamp, for the same reason.

**RoPE.** `_make_rope_tables` mirrors `RopeEmbedder.precompute_freqs_cis`: for axis `i` with
dim `d` and table length `L`, it builds `cos`/`sin` of the `d/2` per-pair angles, stacked as
`[L, d/2, 2]`. These feed `com.microsoft.RotaryEmbedding`, which consumes one cos/sin value per
rotated *pair*, so the tables are `d/2` wide, not `d`. `_identity_position_ids` exists because
the cos/sin caches are already ordered per token — `RotaryEmbedding` must index them with the
identity permutation `[0, 1, ..., seq-1]` to read each token's own row. `_apply_rope` selects
`interleaved=1` (GPT-J pairing `(x[2i], x[2i+1])`, matching this model). When noise-refiner and
context-refiner outputs are unified into one sequence (image tokens first, then caption
tokens), the per-token cos/sin caches are concatenated along the token axis in the same
image-then-caption order, so unified cache row *t* still belongs to unified token *t*.
`_build_position_grids` mirrors `ZImageTransformer2DModel.create_coordinate_grid`: caption
tokens get `(pos=1..cap_len, 0, 0)`; image tokens get `(pos=cap_len+1 [constant],
row=0..H_t-1, col=0..W_t-1)`.

**int4 quantization** (`build()`) quantizes `MatMul` only, never `Gather` — including `Gather`
would int4-quantize the RoPE frequency tables (`model.rope.axis_*_freqs`, the only
initializer-backed Gathers in this diffusion graph, read via `GatherBlockQuantized`),
corrupting positional encoding and degrading image quality.

**Saving.** `save_model` overrides `Model.save_model`, which forces every weight external into
a single `<name>.onnx.data` (`size_threshold_bytes=0`, no sharding). The int4-materialization
and topological-sort steps are mirrored verbatim from the base so quantized builds are
unchanged; only the final write is routed through `save_ir_model_sharded` instead (see External
Data Saving below).

## Text Encoder (`build_text_encoder.py`)

The Z-Image-Turbo pipeline drives its DiT transformer with caption features from a Qwen3
language model. Rather than a full autoregressive decoder, it needs a single-forward encoder
mapping `input_ids`/`attention_mask` to one hidden-state tensor (`encoder_hidden_states`): the
residual stream entering the model's last decoder layer (equivalently, HuggingFace's
`output_hidden_states=True` `hidden_states[-2]`). Unlike `build_transformer.py`/
`build_vae_decoder.py`, this module was never coupled to onnxruntime-genai's `Model` base class
— it builds the graph directly with `onnx.helper`, constructing only `num_hidden_layers - 1`
decoder layers (the last layer, final norm, and lm_head are never built — see
`_build_encoder_graph`'s "tap" step below).

It uses the same fused ops the generic onnxruntime-genai builder emits for a WebGPU int4 Qwen3
`GroupQueryAttention` export: rotary embeddings fused into GQA (`do_rotary=1`, matching the
generic builder's default for any non-DML EP — `base.py`'s `is_fused_rope_supported()` returns
`True` for webgpu), with Q/K per-head RMSNorm kept as separate ops (not fused into GQA, so GQA
stays at its `<=12`-input schema form). GQA derives each token's rotary position internally
from `seqlens_k`/`total_seq_len` — no `position_ids` input exists anywhere in this graph. No KV
cache graph inputs are emitted either; GQA gets empty `past_key`/`past_value`.

`_attention_mask_reformat` derives `seqlens_k`/`total_seq_len` from the 2D `attention_mask`, in
one of two forms selected by `enable_webgpu_graph` (an `--extra_options` flag):
- **Default (`false`)** — the standard Shape-based form WebNN EP prefers (matching `base.py`'s
  `make_attention_mask_reformatting_for_gqa`): `attention_mask -> ReduceSum(axis=1, keepdims=0)
  -> Sub 1 -> Cast(int32) -> seqlens_k`, and `attention_mask -> Shape -> Gather(index 1) ->
  Cast(int32) -> total_seq_len`.
- **`true`** — a `Shape`-free, GPU-only form for WebGPU graph capture: `attention_mask ->
  Cast(int32) -> ReduceSum(axis=1, keepdims=0) -> {Sub 1 -> seqlens_k; ReduceMax ->
  total_seq_len}` (`total_seq_len` comes from a `ReduceMax` over per-row valid token counts
  instead of a `Shape` op).

`_simplified_layernorm` unifies plain `SimplifiedLayerNormalization` (`skip=None`) and the fused
`com.microsoft.SkipSimplifiedLayerNormalization` (`skip=<name>`, computing `sum = x + skip` then
`Y = norm(sum)`) behind one helper. With `need_sum=True`, it also returns the sum (the fused
op's 4th output) — that sum feeds the *next* layer's fused input-norm, or, for the very last
node this module builds, becomes `encoder_hidden_states` directly via `sum_output_name` (no
extra rename op needed). `_build_decoder_layer`'s `root_residual`/`input_ln_skip` parameters
thread this pattern layer to layer: `root_residual` is the residual stream entering a layer's
`input_layernorm` (the embeddings for layer 0, or the previous layer's `resid_before_mlp`
otherwise); `input_ln_skip` is `None` for layer 0, otherwise the previous layer's MLP output to
fuse in as the skip-sum. After the loop, `_build_encoder_graph` "taps" the last layer's
`input_layernorm`: only its residual-sum (4th) output is needed (the normalized-for-attention
output is computed but left unconsumed, since that layer's attention is never run), requiring
only that one layer's `input_layernorm.weight` — a full extra decoder layer is never built.

`dtype` (`"f16"`/`"f32"`) controls I/O and weight dtype throughout; `quantize` controls whether
`MatMul`/`Gather` weights are int4-quantized afterward via `MatMulNBitsQuantizer` — the two are
independent, giving all four `f16`/`f32`/`f16_int4_quant`/`f32_int4_quant` precisions.

## VAE Decoder (`build_vae_decoder.py`)

Graph shape: `conv_in` -> mid block (resnet -> single-head self-attention -> resnet) -> a stack
of up blocks (resnet(s) + optional nearest-2x upsample) -> `conv_norm_out` -> `conv_out`,
mirroring `AutoencoderKL.decoder`'s own structure layer for layer.

GroupNorm has two emission modes, selected by the `fuse_group_norm` extra option:
- **Default (`false`)**: standard-ONNX ops (`Reshape`/`InstanceNormalization`/`Reshape`/`Mul`/
  `Add`), portable to any EP with those kernels.
- **`true`**: the fused `com.microsoft.GroupNorm`/`SkipGroupNorm` contrib ops (NHWC, with the
  residual add and SiLU folded in). Requires an onnxruntime build whose WebGPU EP implements
  these kernels (as of writing, still being brought up there) — the CPU EP has neither.

`-p f16`/`f32` controls I/O and weight dtype throughout — no int4/int8 quantization for this
component, since it's convolution-heavy and not a good `MatMulNBits` target.

## Helper Models (`build_helper_models.py`)

Three tiny, hand-authored graphs (no pretrained weights, no checkpoint input at all) that mirror
the small helper graphs the WebNN JS demo generates to keep intermediate tensors GPU-resident —
but reshaped to this pipeline's `[1, 16, H, W]` hidden_states/sample convention instead of the
WebNN bundle's `[*, 16, 1, H, W]` (no `num_frames` axis here), and with a genuinely float16 I/O
boundary in the f16 variant (not just float16-internal-with-float32-boundary):

- **`scheduler_step_model`**: one flow-matching Euler step, recomputing sigma/sigma_next
  internally from `step_info=[step, num_steps]` (shift=3 schedule, matching the deployed WebNN
  pipeline). `numpy_scheduler_step` is the same math in numpy, used only to verify the exported
  graph against a reference during the build.
- **`vae_pre_process_model`**: VAE scale/shift. Unlike the WebNN bundle
  (`squeeze(axis=2) / scaling_factor + shift_factor`), there's no frame axis to squeeze since
  this pipeline's latents are already `[1, 16, H, W]`.
- **`sc_prep_model`**: resizes the raw VAE-decoded image (`[-1, 1]` range) to CLIP's 224x224
  input and normalizes with the standard OpenAI CLIP mean/std, folding the VAE's
  `[-1,1]->[0,1]` denorm into the per-channel affine (`scale = 0.5/std`,
  `offset = (0.5-mean)/std`). Reverse-engineered from the deployed bundle's
  `sc_prep_model_f16.onnx`: `Resize` with `mode=linear`, `coordinate_transformation_mode=
  asymmetric`, then `Mul`+`Add` by per-channel constants matching these exactly to 4 decimals.
  `numpy_resize_asymmetric_bilinear`/`numpy_sc_prep` are the numpy reference used for build-time
  verification.

Export-time patches, applied after `torch.onnx.export`:
- `_force_dynamic_dim_names`: some torch/opset combinations don't fully honor `dynamic_axes` on
  export, so symbolic dim names are patched into the proto directly as a safety net (mirrors the
  WebNN reference generator's approach).
- `_force_resize_asymmetric`: torch's bilinear interpolate (`align_corners=False`) exports as
  `pytorch_half_pixel`; this forces `asymmetric` instead so it matches the deployed sc_prep
  model exactly (see `sc_prep_model` above).
- `_fix_cast_node_types` (used during `convert_to_f16`): `onnxconverter_common`'s float16
  converter can rewrite a `Cast` node's declared *output* type (`value_info`) to float16
  without updating the node's own `to` attribute, leaving a `Cast(to=FLOAT)` node whose declared
  output is FLOAT16 — onnxruntime rejects that at load time. This patches `to` to match the
  declared output type.
- `convert_to_f16` also un-blocks `Resize` from `onnxconverter_common`'s default
  `op_block_list`. `Resize` is block-listed by default because the ONNX spec hardcodes its
  `scales` input to `tensor(float)` regardless of data dtype, but these graphs only ever drive
  `Resize` via `sizes` (int64) with empty `roi`/`scales`, and ORT's `Resize` kernel natively
  supports float16 `X`/`Y` — so un-blocking it gives a genuinely all-fp16 graph instead of two
  wasted `Cast` nodes wrapped around it.

## Safety Checker (`build_safety_checker.py`)

Unlike the transformer/VAE, this component was never coupled to onnxruntime-genai's `Model`
base class — it's a real pretrained CLIP ViT-L/14 vision classifier (cosine-distance threshold
check against 17 "concept" and 3 "special care" reference embeddings), exported via
`diffusers`' `StableDiffusionSafetyChecker` rather than authored from scratch. Its checkpoint is
a *separate* download from the Z-Image-Turbo checkpoint (`--safety_checker_checkpoint` in
`export_models.py`, or a local `CompVis/stable-diffusion-safety-checker`-compatible folder
passed directly to this script).

`diffusers`' `StableDiffusionSafetyChecker` ships a `forward_onnx(clip_input, images)` method
meant for ONNX export; `SafetyCheckerOnnxWrapper` trims off the `images` masking input/output so
the graph is just `clip_input -> has_nsfw_concepts` — matching the deployed WebNN bundle's
`safety_checker_model_f16.onnx` (confirmed by comparing initializer names/shapes, e.g.
`vision_model.vision_model.embeddings.patch_embedding.weight [1024,3,14,14]`, `concept_embeds
[17,768]`, `special_care_embeds [3,768]`).

Export pipeline: `torch.onnx.export` to f32 (batch pinned to 1, no dynamic axes) ->
`optimize_onnx_graph` (ORT's BASIC-level, EP-neutral optimizations, which can now constant-fold
the `Shape`/`Gather`/`Concat`/`Reshape` machinery `torch.onnx.export` emits for what were
originally dynamic shapes) -> optionally `convert_to_f16` (see Helper Models section above for
what that patches) -> `save_ir_model_sharded`. Optimizing before the f16 conversion means
constant folding runs against the CPU EP's f32 kernels, never a possibly-missing f16 one.

## External data saving (`external_data_utils.py`)

Every exporter saves through `save_ir_model_sharded`, which keeps small initializers
(`<= 1 MiB`) inline in the `.onnx` (so ONNX Runtime graph transformations retain cheap access
to small constants) and writes larger weights to external-data shards capped at 1.9 GiB. That
cap is deliberately just under 2 GiB: each shard has to load into a single JS `ArrayBuffer` in
the browser, and `ArrayBuffer` has a hard 2 GiB (`2**31` byte) ceiling — 1.9 GiB leaves
headroom so a shard's actual on-disk size never crosses it. Output naming:

```
<model>.onnx            # graph + inline weights
<model>.onnx_data       # external shard 0
<model>.onnx_data_1     # external shard 1
<model>.onnx_data_2     # ... (only as many as needed)
```

`onnx_ir.save` does the heavy lifting (streaming, inlining, sharding, and writing each tensor's
location/offset/length so no separate index file is needed), but for >= 2 shards it names them
`<stem>-000i-of-000N<ext>` (1-indexed) rather than the convention above. Since `onnx_ir` also
refuses to overwrite a pre-existing shard file, `save_ir_model_sharded` first clears any stale
output from a previous run, then — after `ir.save` — renames onnx_ir's `-000i-of-000N-` shards
to the `_data[_N]` convention and rewrites the matching `location` strings in the (small)
`.onnx` proto to match. Renaming doesn't touch offsets/lengths, so only the `location` field
needs patching. For exactly one shard, `onnx_ir` already writes the target name directly, so
there's nothing to rename.

## Runtime Pipeline (`run_z_image_turbo.py`)

Unlike the older `../run_z_image_turbo.py` (written against the WebNN demo's bundled models,
which mix WebNN-shaped `[B,16,1,H,W]` 5-frame-axis tensors with dev-model `[1,16,H,W]` ones and
support swapping either in), every model here comes from this repo's own exporters and shares
one fixed shape convention: no frame axis, float16 I/O throughout except the tokenizer/text
encoder boundary. That removes the need for any shape/dtype auto-detection or path-swapping
flags.

Pipeline flow — one box per ONNX model under `<model>/onnx/`, run in this order once per image
(the transformer/scheduler_step pair loops `num_inference_steps` times):

```
prompt
  |
  v
HF tokenizer (<model>/tokenizer/, not ONNX)
  | input_ids, attention_mask
  v
+--------------------------------+
| text_encoder_model_q4f16.onnx  |   input_ids, attention_mask -> encoder_hidden_states
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
```

**Device residency via IO binding.** On a GPU EP, per-step outputs (`latents`/`noise_pred`)
are bound with `bind_output(device_type=...)` so they stay GPU-resident across the denoising
loop instead of round-tripping to host memory between every model call (mirrors the
webnn-developer-preview demo's `useIOBinding` GPU-buffer tensors and this repo's
`ort_playground/py/onnxruntime/llm/llm-ort.py` KV-cache binding). `_run_bound` binds numpy
inputs host-side (`bind_cpu_input` — ORT copies them to the device itself), `OrtValue` inputs
wherever a previous `bind_output` landed (`bind_ortvalue_input`, no copy), and every output via
`bind_output(device_type=...)`.

`to_ort_value` always produces a CPU-backed `OrtValue`: the pip onnxruntime packages don't
support allocating device memory directly from a numpy array for GPU EPs like WebGPU, so actual
device residency only happens through a prior `bind_output` call, never through this method.
`encoder_hidden_states` (the padded prompt embeds) hits this gap — it starts life as a host
numpy buffer, but is fed to the transformer unchanged on every denoising step.
`_build_embeds_uploader` works around it: a trivial in-memory single-node Identity graph (no
declared shape, so any rank/length passes through unchanged; only the dtype is fixed to match
the transformer's `encoder_hidden_states` input) that uploads the embeds to the device once per
prompt via `bind_output`, so they're device-resident for the whole loop instead of re-copied
host->device before each transformer call. The CPU EP doesn't need this — `embeds_uploader`
stays `None` there.

`OrtValue.device_name()` only ever reports `"cpu"` or `"cuda"` — `"cuda"` is a legacy label
onnxruntime's python bindings use for any non-CPU `OrtDevice` (WebGPU/DML/ROCm/CUDA all share
the same generic GPU device-type enum), not an indication it actually ran on CUDA.
`_device_label` relabels it using the EP the pipeline was actually configured with.

**`--sync` and timing.** WebGPU dispatch is asynchronous: `run_with_iobinding` only submits
compute to the GPU queue and returns immediately. `IOBinding.synchronize_outputs()` alone
doesn't force a wait for outputs that stay device-resident — reading each one back to host
does, so that's the only reliable way to get a per-call time that reflects actual compute
instead of just CPU-side submission overhead. This is why `--sync` is off by default (it
serializes every step, each call blocking on the previous step's GPU work) and why per-call
timings are only printed when it's on — without it they'd just measure submission overhead, not
real compute time. `_postprocess`'s `image_ov.numpy()` read is deliberately kept inside the
timed window since it's what forces the final GPU sync regardless of `--sync`; the actual PNG
encode + disk write is excluded, since that's file I/O, not inference. The safety checker's
runtime is measured and printed separately, and deliberately excluded from the pipeline's
total-time metric, matching `../run_z_image_turbo.py`'s `--safety_checker` behavior.
