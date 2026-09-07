# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
"""Shared ONNX external-data save helper for the Z-Image-Turbo exporters.

Keeps small initializers inline in the `.onnx` (so ONNX Runtime graph
transformations retain cheap access to small constants) and writes the larger
weights to size-capped external-data shards named:

    <model>.onnx            # graph + inline (<= size_threshold_bytes) weights
    <model>.onnx_data       # external shard 0
    <model>.onnx_data_1     # external shard 1
    <model>.onnx_data_2     # ... (only as many as needed)

`onnx_ir.save` does the heavy lifting (streams big tensors to disk, keeps small
ones inline, shards at `max_shard_size_bytes`, and writes per-tensor
location/offset/length so no index file is needed). Its only mismatch with the
naming above is the shard filenames it emits for >=2 shards
(`<stem>-000i-of-000N<ext>`); this helper renames those files and rewrites the
matching `location` strings in the (small) `.onnx` proto. Renaming does not
change offsets/lengths, so only the `location` field changes.
"""

import glob
import os
import re

import onnx
import onnx_ir as ir

# External-data layout shared by every Z-Image-Turbo exporter. Small (<= 1 MiB) initializers
# stay inline in the `.onnx` so ONNX Runtime graph transformations keep cheap access to small
# constants; larger weights go to size-capped `.onnx_data[_N]` shards. The shard cap sits just
# under 2 GiB: each shard loads into a single JS ArrayBuffer in the browser, and ArrayBuffer has
# a hard 2 GiB (2**31 byte) ceiling, so 1.9 GiB leaves headroom to avoid that bottleneck.
INLINE_SIZE_THRESHOLD_BYTES = 1 * 1024**2
MAX_SHARD_SIZE_BYTES = int(1.9 * 1024**3)

# onnx_ir shard filename convention: "<stem>-000i-of-000N<ext>" (i, N 1-indexed).
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})")


def save_ir_model_sharded(
    model, out_dir, onnx_filename, *, size_threshold_bytes, max_shard_size_bytes, callback=None
):
    """Save an onnx_ir `model` with small weights inline and external data sharded.

    Args:
        model: the `onnx_ir.Model` to save.
        out_dir: directory to write the `.onnx` and `.onnx_data*` files to.
        onnx_filename: the `.onnx` filename (e.g. "transformer_model_q4f16.onnx").
        size_threshold_bytes: initializers larger than this go external; the rest stay inline.
        max_shard_size_bytes: maximum on-disk size of each external-data shard file.
        callback: optional per-tensor `onnx_ir.save` callback (progress/logging).

    Returns:
        The path to the written `.onnx` file.
    """
    out_path = os.path.join(out_dir, onnx_filename)
    data_base = onnx_filename + "_data"  # e.g. "...q4f16.onnx" -> "...q4f16.onnx_data"
    stem, ext = os.path.splitext(data_base)  # ("...q4f16", ".onnx_data")

    # Clear stale outputs so reruns overwrite cleanly (onnx_ir's sharded writer refuses
    # to overwrite a pre-existing shard file) -- final names and any leftover temp shards.
    stale = [out_path, *glob.glob(os.path.join(out_dir, data_base + "*")),
             *glob.glob(os.path.join(out_dir, f"{stem}-*-of-*{ext}"))]
    for path in stale:
        if os.path.exists(path):
            os.remove(path)

    ir.save(
        model, out_path, external_data=data_base,
        size_threshold_bytes=size_threshold_bytes, max_shard_size_bytes=max_shard_size_bytes,
        callback=callback,
    )

    # For >=2 shards onnx_ir wrote "<stem>-000i-of-000N<ext>"; for exactly one it wrote
    # "<data_base>" (already our target name -> nothing to rename or patch).
    temp_shards = sorted(glob.glob(os.path.join(out_dir, f"{stem}-*-of-*{ext}")))
    if not temp_shards:
        return out_path

    remap = {}  # onnx_ir shard basename -> final basename
    for path in temp_shards:
        base = os.path.basename(path)
        idx = int(_SHARD_RE.search(base).group(1))  # 1-based shard index
        final = data_base if idx == 1 else f"{data_base}_{idx - 1}"
        os.replace(path, os.path.join(out_dir, final))
        remap[base] = final

    # Rewrite the per-tensor `location` in the (small) proto to the renamed files.
    proto = onnx.load(out_path, load_external_data=False)
    for init in proto.graph.initializer:
        for kv in init.external_data:
            if kv.key == "location" and kv.value in remap:
                kv.value = remap[kv.value]
    onnx.save(proto, out_path)
    return out_path
