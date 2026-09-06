# Z-Image-Turbo Standalone Exporters

Self-contained experiment: export pieces of the
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) text-to-image pipeline to
plain ONNX graphs. Unlike `../builders/zimage*.py`, these `build_*.py` scripts depend only on
the *public* `onnxruntime-genai` pip package (not this repo's own `../builders/base.py` source
tree), so each one runs standalone with just `pip install -r requirements.txt` -- no
onnxruntime-genai checkout required.

`export_models.py` is the main entry point; each component also has its own standalone CLI.

## Status

| Component | Script | Status |
|---|---|---|
| Transformer trunk | `build_transformer.py` | done |
| VAE decoder | `build_vae_decoder.py` | done |
| Helper models (scheduler_step / vae_pre_process / sc_prep) | `build_helper_models.py` | done |
| Safety checker | `build_safety_checker.py` | done (needs its own separate checkpoint, see below) |
| Text encoder | `build_text_encoder.py` | not yet ported |

For the not-yet-ported text encoder, use `../build_z_image_turbo.py` (the version coupled to
this repo's own `../builders/` tree) in the meantime.

## Install

```bash
cd src/python/py/models/z-image-turbo
pip install -r requirements.txt  # needs Python 3.11/3.12/3.13
```

## Usage

Download the Z-Image-Turbo checkpoint first:

```py
from huggingface_hub import snapshot_download
snapshot_download("Tongyi-MAI/Z-Image-Turbo", local_dir="path_to_local_folder")
```

Then export everything at once (default `-m all`, default output dir `z-image-turbo-onnx/`):

```bash
python export_models.py path_to_local_folder
```

Or one component at a time:

```bash
python export_models.py path_to_local_folder -m transformer -o my_output_dir
python export_models.py path_to_local_folder -m vae_decoder
python export_models.py path_to_local_folder -m helper_models
```

`path_to_local_folder` may be the checkpoint's repo root (with `transformer/`/`vae/`
subfolders) or a component subfolder directly -- both are auto-detected.

The safety checker is a real pretrained CLIP classifier from a *separate* checkpoint
(`CompVis/stable-diffusion-safety-checker`, not part of the Z-Image-Turbo checkpoint), so it
needs its own `--safety_checker_checkpoint`:

```py
from huggingface_hub import snapshot_download
snapshot_download("CompVis/stable-diffusion-safety-checker", local_dir="path_to_safety_checker_folder")
```

```bash
python export_models.py path_to_local_folder -m safety_checker --safety_checker_checkpoint path_to_safety_checker_folder
```

`-m all` includes it too if `--safety_checker_checkpoint` is given; otherwise it's skipped
(with a message) so `-m all` still works without it.

Each `build_*.py` also works standalone, e.g.:

```bash
python build_transformer.py path_to_local_folder/transformer -o my_output_dir -p f16_int4_quant
python build_vae_decoder.py path_to_local_folder/vae -o my_output_dir -p f16
python build_helper_models.py -o my_output_dir -p f16
python build_safety_checker.py path_to_safety_checker_folder -o my_output_dir -p f16
```

All `.onnx` output (+ external data) lands under `<output_dir>/onnx/`, so multiple components
can share one output directory. No `genai_config.json` is produced -- these are standalone
ONNX graphs, not onnxruntime-genai C++ runtime integrations. A caller drives the diffusion
sampling loop itself; see `run_z_image_turbo.py` for a reference driver.

`export_models.py` (but not the individual `build_*.py` scripts) also copies the checkpoint's
tokenizer files (`merges.txt`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json` -- from
the checkpoint's `tokenizer/` folder, resolved relative to `text_encoder/`) into
`<output_dir>/tokenizer/`, so the output directory is self-contained.

### Options

- `-p/--precision`: `build_transformer.py` supports `f16` / `f32` / `f16_int4_quant` (default)
  / `f32_int4_quant`; `build_vae_decoder.py`, `build_helper_models.py`, and
  `build_safety_checker.py` support `f16` (default) / `f32` only.
- `--extra_options key=value ...`: passed through to the component's builder, e.g.
  `fuse_group_norm=true` (VAE decoder) or `height=512 width=512 num_inference_steps=8`
  (helper models).
