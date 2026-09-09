# Z-Image-Turbo Standalone Exporters

Self-contained experiment: export pieces of the
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) text-to-image pipeline to
plain ONNX graphs, using only the _public_ `onnxruntime-genai` pip package -- no checkout of
this repo's own source tree required, just `pip install -r requirements.txt`.

`export_models.py` is the entry point for all components.

## Status

| Component                                                    | Status                                              |
| ------------------------------------------------------------- | ---------------------------------------------------- |
| Transformer trunk                                              | done                                                 |
| Text encoder (Qwen3)                                           | done                                                 |
| VAE decoder                                                    | done                                                 |
| Helper models (scheduler_step / vae_pre_process / sc_prep)     | done                                                 |
| Safety checker                                                 | done (needs its own separate checkpoint, see below) |

Every component is ported -- `export_models.py -m all` (or no `-m`) builds the full pipeline
in one bundle directory (safety_checker needs `--safety_checker_checkpoint`, see below; it's
skipped otherwise).

## Install

Python >= 3.12.0 is recommended. Create and activate a virtual environment first, e.g.:

```bash
cd src/python/py/models/z-image-turbo
py -3.12 -m venv .venv
source .venv/Scripts/activate  # on Windows (bash); use `.venv\Scripts\activate` in cmd/PowerShell, or `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
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
python export_models.py path_to_local_folder -m text_encoder
python export_models.py path_to_local_folder -m vae_decoder
python export_models.py path_to_local_folder -m helper_models
```

`path_to_local_folder` may be the checkpoint's repo root (with `transformer/`/`text_encoder/`/
`vae/` subfolders) or a component subfolder directly -- both are auto-detected.

The safety checker is a real pretrained CLIP classifier from a _separate_ checkpoint
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

All `.onnx` output (+ external data) lands under `<output_dir>/onnx/`, so multiple components
can share one output directory. No `genai_config.json` is produced -- these are standalone
ONNX graphs, not onnxruntime-genai C++ runtime integrations. A caller drives the diffusion
sampling loop itself.

`export_models.py` (but not the individual `build_*.py` scripts) also copies the checkpoint's
tokenizer files (`merges.txt`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json` -- from
the checkpoint's `tokenizer/` folder, resolved relative to `text_encoder/`) into
`<output_dir>/tokenizer/`, so the output directory is self-contained.

### Options

- `-p/--precision`: for `-m transformer`/`-m text_encoder`, `f16` / `f32` / `f16_int4_quant`
  (default) / `f32_int4_quant`; for `-m vae_decoder`/`-m helper_models`/`-m safety_checker`,
  `f16` (default) / `f32` only.
- `--extra_options key=value ...`: passed through to the component's builder, e.g.
  `fuse_group_norm=true` (`-m vae_decoder`) or `height=512 width=512 num_inference_steps=8`
  (`-m helper_models`).
