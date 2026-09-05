# run_z_image_turbo.py changelog

## 2026-09-05 -- WebGPU/Windows fixes, IO-bound GPU residency

Found while testing on Windows against a real WebGPU device.

### Fixes

- **WebGPU `Can't allocate memory on the device` crash.** `to_ort_value()` called
  `OrtValue.ortvalue_from_numpy(array, device_type="webgpu", device_id=gpu)`, which tries to
  allocate device memory directly from a numpy buffer -- unsupported by the pip `onnxruntime`
  WebGPU build, so every per-step tensor creation crashed immediately. `to_ort_value()` now
  always creates a plain CPU `OrtValue`; device residency is achieved via IO binding instead
  (see below), not by forcing input allocation onto the device.
- **Windows console `UnicodeEncodeError`.** The default prompt contains Chinese characters,
  which crashed `print(f"Prompt:\n{prompt}")` on a cp1252 Windows console. Fixed by
  reconfiguring stdout in `main()`: `sys.stdout.reconfigure(errors="backslashreplace")`.
- **`ModuleNotFoundError: resource`.** `peak_memory_mb()` used the POSIX-only `resource`
  module (plus `os.uname()`), neither of which exist on Windows. Replaced with `psutil`
  (`peak_wset` on Windows, `rss` elsewhere); added to `requirements.txt`.

### IO binding for GPU-resident tensors across the denoising loop

Adopted the pattern from `ort_playground/py/onnxruntime/llm/llm-ort.py`'s
`bind_output`/`bind_ortvalue_input`/`bind_cpu_input` usage:

- Every session (`text_encoder`, `transformer`, `scheduler_step`, `vae_pre_process`,
  `vae_decoder`) now owns one persistent `io_binding()`, created once in
  `ZImagePipeline.__init__` and reused across calls.
- `_run_bound()`: numpy inputs go through `bind_cpu_input` (ORT copies host->device itself);
  `OrtValue` inputs chained from a previous call's output go through `bind_ortvalue_input`
  (no copy); every output is bound with `bind_output(device_type=self.device_type,
  device_id=self.device_id)`, so ORT allocates it directly on the target device instead of
  the host allocation `run_with_ort_values` used previously.
- `_run_transformer` / `_run_scheduler_step` / `_run_vae_pre_process` / `_run_vae_decoder` /
  `encode_prompt`'s text-encoder call all run through `_run_bound` + `run_with_iobinding`.

**Effect:** `noise_pred` / `latents` / `scaled_latents` stay device-resident across the
denoising loop instead of round-tripping to host memory every step -- confirmed by the IO
device log (below) reporting `webgpu` instead of `cpu` for these tensors after the first step.

Tensors that stay `cpu` regardless, given this pipeline's exported graphs:

| Tensor | Why it can't stay on-device |
|---|---|
| `timestep` / `step_info` | Tiny per-step scalars rebuilt fresh every iteration anyway (`llm-ort.py`'s decode loop does the same, keeping only the large KV buffers device-resident). |
| `input_ids` / `attention_mask` | Tiny, used once per run. |
| `vae_decoder`'s `latent_sample` | `vae_pre_process` outputs float16 but `vae_decoder` wants float32; the exported graph has no on-device Cast op, and the pip webgpu package can't allocate device memory straight from a numpy array (same limitation as the crash above). |
| `encoder_hidden_states` | Re-uploaded on every transformer call: `IOBinding` can only capture *outputs* on-device, and no exported graph echoes this input back out as an output to relocate it once (unlike `llm-ort.py`'s KV cache). Fixing this would need an identity/cast output added to `build_transformer.py`'s export. |

### Per-call IO device logging

`_log_io()` (called from `_run_bound`, always on -- there is no `--verbose` flag) prints every
input/output tensor name tagged `in`/`out` with its device, e.g.:

```
  [transformer] io:
    in  hidden_states: webgpu
    in  timestep: cpu
    in  encoder_hidden_states: cpu
    out sample: webgpu
```

`OrtValue.device_name()` only ever reports `"cpu"` or `"cuda"` -- `"cuda"` is a legacy label
onnxruntime's python bindings use for any non-CPU `OrtDevice` (WebGPU/DML/ROCm/CUDA all share
the same generic GPU device-type enum), not an indication anything ran on CUDA.
`_device_label()` relabels it using the EP the pipeline was actually configured with
(`self.device_type`), so the log reads `webgpu` instead of the misleading `cuda`. This replaced
the old one-off `print(f"  tensor location: noise_pred={...}, latents={...}")` line in `run()`.

### WebGPU device listing readability

`select_webgpu_device()`'s adapter listing printed one crammed line per device (`[0] Intel
Corporation vendor_id=0x8086 device_id=0xb080 {'Description': ..., ...}`). Reformatted to one
field per line.
