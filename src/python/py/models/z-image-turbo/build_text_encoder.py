# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# --------------------------------------------------------------------------
"""Build the Z-Image-Turbo text encoder ONNX graph directly from a Qwen3 HF checkpoint.

Self-contained port of ../builders/zimage_text_encoder.py: that module already has no
dependency on this repo's generic model-builder infrastructure (`Model`/`base.py`) in the
first place -- it builds the graph directly with `onnx.helper` -- so this file only adds the
`build()`/CLI wrapper matching the other standalone build_*.py scripts' conventions. Only
`numpy`, `onnx`, `torch`, `transformers`, and `onnxruntime` (for
`onnxruntime.quantization.matmul_nbits_quantizer`) are needed.

The Z-Image-Turbo pipeline drives its DiT transformer with caption features taken
from a Qwen3 language model. Rather than a full autoregressive decoder, it needs a
single-forward encoder that maps `input_ids`/`attention_mask` to one hidden-state
tensor (`encoder_hidden_states`): the residual stream entering the model's last
decoder layer (equivalently, HuggingFace's `output_hidden_states=True`
`hidden_states[-2]`).

This module builds that graph directly with `onnx.helper`. It constructs only
`num_hidden_layers - 1` decoder layers (the last layer, final norm, and lm_head are
never built), using the same fused ops the generic builder emits for a WebGPU int4
Qwen3 GroupQueryAttention export: `com.microsoft.GroupQueryAttention` with rotary
embeddings fused in (`do_rotary=1`, matching the generic builder's default for any
non-DML EP), and Q/K per-head RMSNorm kept as separate ops (not fused into GQA, so GQA
stays at its <=12-input schema form). GQA derives each token's rotary position
internally from `seqlens_k`/`total_seq_len` (itself derived from `attention_mask` via a
reformatting subgraph) -- no `position_ids` input exists anywhere in this graph, external or
internal. The mask reformat defaults to the standard `Shape`-based form (matching base.py's
`make_attention_mask_reformatting_for_gqa`), which WebNN EP prefers; passing
`enable_webgpu_graph=true` via `--extra_options` instead emits the Shape-free, GPU-only
graph-capture variant for WebGPU graph capture. No KV cache graph inputs are ever emitted;
GroupQueryAttention gets empty `past_key`/`past_value`.

`dtype` ("f16" or "f32") controls the I/O and weight dtype throughout; `quantize`
controls whether MatMul/Gather weights are int4-quantized afterward -- the two are
independent, giving all four `f16`/`f32`/`f16_int4_quant`/`f32_int4_quant` precisions.
"""

import argparse
import json
import os

import numpy as np
import onnx
import onnx_ir as ir
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, QuantFormat
from transformers import AutoModelForCausalLM

from external_data_utils import (
    INLINE_SIZE_THRESHOLD_BYTES,
    MAX_SHARD_SIZE_BYTES,
    save_ir_model_sharded,
)

DEFAULT_OUTPUT_DIR = "z-image-turbo-onnx"

# Filename suffix per precision, matching ../build_z_image_turbo.py's TEXT_ENCODER_PRECISIONS
# and the WebNN bundle's own naming convention (text_encoder_model_q4f16.onnx).
PRECISION_CONFIGS = {
    "f16": {"dtype": "f16", "quantize": False, "suffix": "f16"},
    "f32": {"dtype": "f32", "quantize": False, "suffix": "f32"},
    "f16_int4_quant": {"dtype": "f16", "quantize": True, "suffix": "q4f16"},
    "f32_int4_quant": {"dtype": "f32", "quantize": True, "suffix": "q4f32"},
}


def parse_extra_options(pairs):
    """Parse `key=value` strings (as passed via `--extra_options`) into a dict."""
    extra_options = {}
    for kv_str in pairs or []:
        key, _, value = kv_str.partition("=")
        extra_options[key.strip()] = value.strip()
    return extra_options


def _read_qwen3_config(text_encoder_dir):
    with open(os.path.join(text_encoder_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "num_layers": cfg["num_hidden_layers"],
        "hidden_size": cfg["hidden_size"],
        "num_attn_heads": cfg["num_attention_heads"],
        "num_kv_heads": cfg["num_key_value_heads"],
        "head_size": cfg["head_dim"],
        "rope_theta": float(cfg["rope_theta"]),
        "rms_norm_eps": float(cfg["rms_norm_eps"]),
        "max_position_embeddings": cfg["max_position_embeddings"],
    }


_TORCH_DTYPES = {"f16": torch.float16, "f32": torch.float32}
_ONNX_DTYPES = {"f16": TensorProto.FLOAT16, "f32": TensorProto.FLOAT}
_NP_DTYPES = {"f16": np.float16, "f32": np.float32}


def _rotary_cos_sin_tables(head_size, rope_theta, max_position_embeddings, np_dtype):
    # Standard (non-scaled) Qwen3 RoPE: Z-Image-Turbo's checkpoint has rope_scaling=null.
    dim = head_size
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, dim, 2, dtype=np.int64).astype(np.float64) / dim))
    t = np.arange(max_position_embeddings, dtype=np.float64)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    # com.microsoft.RotaryEmbedding expects cos/sin caches of shape [max_seq_len, head_size/2]
    cos_cache = np.cos(emb)[:, : dim // 2].astype(np_dtype)
    sin_cache = np.sin(emb)[:, : dim // 2].astype(np_dtype)
    return cos_cache, sin_cache


class _GraphBuilder:
    """Accumulates onnx NodeProtos and initializers for the encoder graph."""

    def __init__(self):
        self.nodes = []
        self.initializers = []

    def initializer(self, name, array):
        array = np.asarray(array)
        # np.ascontiguousarray promotes 0-d scalars to shape (1,); keep true scalars 0-d so a
        # scalar Gather index yields a scalar Gather output. GroupQueryAttention's
        # total_sequence_length must be a scalar (this is the form the generic builder emits and
        # that WebNN EP's static-shape GQA requires); a 1-D [1] length trips it up.
        if array.ndim:
            array = np.ascontiguousarray(array)
        self.initializers.append(numpy_helper.from_array(array, name=name))
        return name

    def const_i64(self, name, values):
        return self.initializer(name, np.array(values, dtype=np.int64))

    def node(self, op_type, inputs, outputs, name, domain="", **attrs):
        self.nodes.append(helper.make_node(op_type, inputs, outputs, name=name, domain=domain, **attrs))
        return outputs[0]


def _attention_mask_reformat(gb, attention_mask_name, enable_webgpu_graph=False):
    # Derive GroupQueryAttention's seqlens_k/total_seq_len from the 2D attention_mask. Two
    # variants, matching the generic builder; selected by `enable_webgpu_graph`:
    #
    # enable_webgpu_graph=True -- No Shape op, so every op runs on GPU (required
    #   for WebGPU graph capture); total_seq_len comes from ReduceMax over the per-row valid
    #   token counts:
    #     attention_mask -> Cast(int32) -> ReduceSum(axis=1, keepdims=0) -> {Sub 1 -> seqlens_k;
    #                                                                        ReduceMax -> total_seq_len}
    #
    # enable_webgpu_graph=False (default) -- total_seq_len is the (padded) sequence-length dim via a Shape op (runs on CPU).
    #   This is the form WebNN EP prefers:
    #     attention_mask -> ReduceSum(axis=1, keepdims=0) -> Sub 1 -> Cast(int32) -> seqlens_k
    #     attention_mask -> Shape -> Gather(index 1) -> Cast(int32) -> total_seq_len
    if enable_webgpu_graph:
        mask_i32 = gb.node("Cast", [attention_mask_name], ["attn_mask_reformat/mask_i32"],
                            "attn_mask_reformat/Cast", to=TensorProto.INT32)
        axis1 = gb.const_i64("attn_mask_reformat/axis1", [1])
        mask_sum = gb.node("ReduceSum", [mask_i32, axis1], ["attn_mask_reformat/mask_sum"],
                            "attn_mask_reformat/ReduceSum", keepdims=0)
        one_i32 = gb.initializer("attn_mask_reformat/one_i32", np.array([1], dtype=np.int32))
        seqlens_k = gb.node("Sub", [mask_sum, one_i32], ["attn_mask_reformat/seqlens_k"],
                             "attn_mask_reformat/Sub")
        total_seq_len = gb.node("ReduceMax", [mask_sum], ["attn_mask_reformat/total_seq_len"],
                                 "attn_mask_reformat/ReduceMax", keepdims=0)
        return seqlens_k, total_seq_len

    # Standard (Shape-based) form.
    axis1 = gb.const_i64("attn_mask_reformat/axis1", [1])
    mask_sum = gb.node("ReduceSum", [attention_mask_name, axis1], ["attn_mask_reformat/mask_sum"],
                        "attn_mask_reformat/ReduceSum", keepdims=0)
    one_i64 = gb.const_i64("attn_mask_reformat/one_i64", [1])
    sub = gb.node("Sub", [mask_sum, one_i64], ["attn_mask_reformat/sub"], "attn_mask_reformat/Sub")
    seqlens_k = gb.node("Cast", [sub], ["attn_mask_reformat/seqlens_k"],
                        "attn_mask_reformat/Sub/Cast", to=TensorProto.INT32)

    shape = gb.node("Shape", [attention_mask_name], ["attn_mask_reformat/shape"], "attn_mask_reformat/Shape")
    idx1 = gb.const_i64("attn_mask_reformat/idx1", 1)
    gathered = gb.node("Gather", [shape, idx1], ["attn_mask_reformat/gathered"],
                        "attn_mask_reformat/Gather", axis=0)
    total_seq_len = gb.node("Cast", [gathered], ["attn_mask_reformat/total_seq_len"],
                            "attn_mask_reformat/Gather/Cast", to=TensorProto.INT32)
    return seqlens_k, total_seq_len


def _simplified_layernorm(gb, x, weight_name, eps, name_prefix, skip=None, need_sum=False, sum_output_name=None):
    # skip=None -> plain SimplifiedLayerNormalization (default "" domain, axis/stash_type attrs).
    # skip=<name> -> SkipSimplifiedLayerNormalization (com.microsoft domain): computes
    # sum = x + skip, then Y = norm(sum). need_sum=True also returns the sum (4th output);
    # the sum feeds the *next* layer's fused input-norm, or -- for the very last node this
    # module builds -- becomes `encoder_hidden_states` directly (via sum_output_name, so the
    # graph's final output tensor is produced directly by this node, no extra rename op).
    inputs = [x, weight_name] if skip is None else [x, skip, weight_name]
    op_type = ("Skip" if skip is not None else "") + "SimplifiedLayerNormalization"
    domain = "com.microsoft" if skip is not None else ""
    y = f"{name_prefix}/Y"
    sum_name = sum_output_name or f"{name_prefix}/sum"
    outputs = [y] if skip is None else [y, "", "", (sum_name if need_sum else "")]
    attrs = {"epsilon": eps}
    if skip is None:
        attrs.update(axis=-1, stash_type=1)
    gb.node(op_type, inputs, outputs, name_prefix, domain=domain, **attrs)
    return y, (outputs[3] if skip is not None else None)


def _qk_head_norm(gb, x, weight_name, num_heads, head_size, eps, name_prefix):
    # BxSxD -> Bx(S*N)xH -> SimplifiedLayerNormalization -> BxSxD
    shape1 = gb.const_i64(f"{name_prefix}/shape1", [0, -1, head_size])
    r1 = gb.node("Reshape", [x, shape1], [f"{name_prefix}/Reshape_1/out"], f"{name_prefix}/Reshape_1")
    normed, _ = _simplified_layernorm(gb, r1, weight_name, eps, f"{name_prefix}/SimplifiedLayerNormalization")
    shape2 = gb.const_i64(f"{name_prefix}/shape2", [0, -1, num_heads * head_size])
    return gb.node("Reshape", [normed, shape2], [f"{name_prefix}/Reshape_2/out"], f"{name_prefix}/Reshape_2")


def _matmul(gb, x, weight_param, name_prefix, torch_dtype):
    # weight_param: an HF nn.Linear weight, shape [out_features, in_features]. ONNX MatMul
    # needs [in_features, out_features], so transpose. Qwen3 has no bias on any projection
    # (attention_bias=false, and the MLP/output projections don't use bias either).
    weight = weight_param.detach().to(torch_dtype).numpy().T
    weight_name = gb.initializer(f"{name_prefix}.weight", weight)
    return gb.node("MatMul", [x, weight_name], [f"{name_prefix}/out"], f"{name_prefix}/MatMul")


def _build_decoder_layer(gb, layer_id, layer, root_residual, input_ln_skip, dims,
                          seqlens_k_name, total_seq_len_name, cos_cache_name, sin_cache_name, torch_dtype):
    """Emits one full Qwen3 decoder layer (attention + MLP).

    root_residual: the residual stream entering this layer's input_layernorm (== the
        embeddings output for layer 0, or the previous layer's `resid_before_mlp` otherwise).
    input_ln_skip: None for layer 0 (plain SimplifiedLayerNormalization); otherwise the
        previous layer's MLP output, fused into this layer's input_layernorm as a
        SkipSimplifiedLayerNormalization (this is the "residual add from the previous layer,
        fused into this layer's norm" pattern the generic builder also uses).

    Returns (resid_before_mlp, mlp_output) -- both needed to build the next layer, and (for
    the last layer this module builds) resid_before_mlp/mlp_output together are what the
    caller feeds into one more `_simplified_layernorm(..., skip=mlp_output, need_sum=True)`
    call (using the *next* layer's input_layernorm weight) to produce `encoder_hidden_states`.
    """
    num_attn_heads, num_kv_heads = dims["num_attn_heads"], dims["num_kv_heads"]
    head_size, eps = dims["head_size"], dims["rms_norm_eps"]

    ln1_w = gb.initializer(
        f"model.layers.{layer_id}.input_layernorm.weight",
        layer.input_layernorm.weight.detach().to(torch_dtype).numpy(),
    )
    normed, resid_before_attn = _simplified_layernorm(
        gb, root_residual, ln1_w, eps, f"layer{layer_id}/input_layernorm",
        skip=input_ln_skip, need_sum=(input_ln_skip is not None),
    )
    if input_ln_skip is None:
        resid_before_attn = root_residual

    attn = layer.self_attn
    q = _matmul(gb, normed, attn.q_proj.weight, f"layer{layer_id}/attn/q_proj", torch_dtype)
    k = _matmul(gb, normed, attn.k_proj.weight, f"layer{layer_id}/attn/k_proj", torch_dtype)
    v = _matmul(gb, normed, attn.v_proj.weight, f"layer{layer_id}/attn/v_proj", torch_dtype)

    qn_w = gb.initializer(f"model.layers.{layer_id}.attn.q_norm.weight",
                           attn.q_norm.weight.detach().to(torch_dtype).numpy())
    kn_w = gb.initializer(f"model.layers.{layer_id}.attn.k_norm.weight",
                           attn.k_norm.weight.detach().to(torch_dtype).numpy())
    q = _qk_head_norm(gb, q, qn_w, num_attn_heads, head_size, eps, f"layer{layer_id}/attn/q_norm")
    k = _qk_head_norm(gb, k, kn_w, num_kv_heads, head_size, eps, f"layer{layer_id}/attn/k_norm")

    # RotaryEmbedding is fused into GroupQueryAttention (do_rotary=1) rather than emitted as
    # separate nodes -- matching the generic builder's default for any non-DML EP
    # (base.py's is_fused_rope_supported() returns True for webgpu). GQA derives each token's
    # rotary position internally from seqlens_k/total_seq_len -- its position_ids input is
    # always left empty, fused or not.
    scale = float(1.0 / np.sqrt(head_size))
    attn_out = gb.node(
        "GroupQueryAttention",
        [q, k, v, "", "", seqlens_k_name, total_seq_len_name, cos_cache_name, sin_cache_name, "", "", ""],
        [f"layer{layer_id}/attn/gqa_out", f"layer{layer_id}/attn/present_k", f"layer{layer_id}/attn/present_v"],
        f"layer{layer_id}/attn/GQA", domain="com.microsoft",
        num_heads=num_attn_heads, kv_num_heads=num_kv_heads, scale=scale,
        local_window_size=-1, do_rotary=1, rotary_interleaved=0,
    )
    o = _matmul(gb, attn_out, attn.o_proj.weight, f"layer{layer_id}/attn/o_proj", torch_dtype)

    post_ln_w = gb.initializer(
        f"model.layers.{layer_id}.post_attention_layernorm.weight",
        layer.post_attention_layernorm.weight.detach().to(torch_dtype).numpy(),
    )
    mlp_in, resid_before_mlp = _simplified_layernorm(
        gb, resid_before_attn, post_ln_w, eps, f"layer{layer_id}/post_attention_layernorm",
        skip=o, need_sum=True,
    )

    mlp = layer.mlp
    gate = _matmul(gb, mlp_in, mlp.gate_proj.weight, f"layer{layer_id}/mlp/gate_proj", torch_dtype)
    up = _matmul(gb, mlp_in, mlp.up_proj.weight, f"layer{layer_id}/mlp/up_proj", torch_dtype)
    sig = gb.node("Sigmoid", [gate], [f"layer{layer_id}/mlp/sigmoid"], f"layer{layer_id}/mlp/Sigmoid")
    silu = gb.node("Mul", [gate, sig], [f"layer{layer_id}/mlp/silu"], f"layer{layer_id}/mlp/Mul_silu")
    gated = gb.node("Mul", [silu, up], [f"layer{layer_id}/mlp/gated"], f"layer{layer_id}/mlp/Mul_gated")
    down = _matmul(gb, gated, mlp.down_proj.weight, f"layer{layer_id}/mlp/down_proj", torch_dtype)

    return resid_before_mlp, down


def _quantize_int4(onnx_model):
    """Quantize the encoder graph to int4 using MatMulNBitsQuantizer.

    Applies int4 weight-only quantization to MatMul and Gather ops:
    - MatMul -> MatMulNBits
    - Gather (embedding) -> GatherBlockQuantized
    """
    quantizer = MatMulNBitsQuantizer(
        model=onnx_model,
        bits=4,
        block_size=32,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=("MatMul", "Gather"),
    )
    quantizer.process()
    quantized = quantizer.model.model
    # MatMulNBitsQuantizer updates opset to 21 for int4 support, which requires IR version >= 10
    quantized.ir_version = 10
    return quantized


def _build_encoder_graph(checkpoint_dir, dtype="f16", enable_webgpu_graph=False):
    torch_dtype = _TORCH_DTYPES[dtype]
    onnx_dtype = _ONNX_DTYPES[dtype]
    np_dtype = _NP_DTYPES[dtype]

    dims = _read_qwen3_config(checkpoint_dir)
    num_layers = dims["num_layers"]

    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    model.eval()
    layers = model.model.layers

    gb = _GraphBuilder()
    input_ids_name, attn_mask_name = "input_ids", "attention_mask"

    cos_cache, sin_cache = _rotary_cos_sin_tables(
        dims["head_size"], dims["rope_theta"], dims["max_position_embeddings"], np_dtype
    )
    cos_cache_name = gb.initializer("cos_cache", cos_cache)
    sin_cache_name = gb.initializer("sin_cache", sin_cache)

    embed_w = gb.initializer("model.embed_tokens.weight",
                              model.model.embed_tokens.weight.detach().to(torch_dtype).numpy())
    embeddings = gb.node("Gather", [embed_w, input_ids_name], ["embeddings"], "embed/Gather", axis=0)

    seqlens_k_name, total_seq_len_name = _attention_mask_reformat(gb, attn_mask_name, enable_webgpu_graph)

    root_residual, input_ln_skip = embeddings, None
    for layer_id in range(num_layers - 1):
        resid_before_mlp, mlp_out = _build_decoder_layer(
            gb, layer_id, layers[layer_id], root_residual, input_ln_skip, dims,
            seqlens_k_name, total_seq_len_name, cos_cache_name, sin_cache_name, torch_dtype,
        )
        root_residual, input_ln_skip = resid_before_mlp, mlp_out

    # Tap the last layer's input_layernorm: only its residual-sum (4th) output is needed --
    # the normalized-for-attention output (Y) is computed but left unconsumed, since we never
    # run that layer's attention. Requires only that one layer's input_layernorm.weight.
    tap_id = num_layers - 1
    tap_ln_w = gb.initializer(
        f"model.layers.{tap_id}.input_layernorm.weight",
        layers[tap_id].input_layernorm.weight.detach().to(torch_dtype).numpy(),
    )
    _, encoder_hidden_states = _simplified_layernorm(
        gb, root_residual, tap_ln_w, dims["rms_norm_eps"], f"layer{tap_id}/input_layernorm",
        skip=input_ln_skip, need_sum=True, sum_output_name="encoder_hidden_states",
    )
    assert encoder_hidden_states is not None  # guaranteed by need_sum=True with a non-None skip

    # batch is fixed at 1 across the whole pipeline (the transformer hardcodes batch=1), so
    # pin it here too -- a static leading dim helps ORT's graph optimizations.
    graph_inputs = [
        helper.make_tensor_value_info(input_ids_name, TensorProto.INT64, [1, "sequence_length"]),
        helper.make_tensor_value_info(attn_mask_name, TensorProto.INT64, [1, "total_sequence_length"]),
    ]
    graph_output = helper.make_tensor_value_info(
        encoder_hidden_states, onnx_dtype, [1, "sequence_length", dims["hidden_size"]]
    )
    graph = helper.make_graph(gb.nodes, "zimage_text_encoder", graph_inputs, [graph_output], initializer=gb.initializers)
    onnx_model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    onnx_model.ir_version = 9
    return onnx_model


def export_qwen3_text_encoder(checkpoint_dir, output_onnx_path, quantize, dtype="f16", enable_webgpu_graph=False):
    onnx_model = _build_encoder_graph(checkpoint_dir, dtype=dtype, enable_webgpu_graph=enable_webgpu_graph)
    if quantize:
        onnx_model = _quantize_int4(onnx_model)

    out_dir = os.path.dirname(os.path.abspath(output_onnx_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    # Same save path as the transformer: small (<= 1 MiB) weights inline for ORT graph
    # transformations, larger weights sharded into `< 1.9 GiB` `.onnx_data[_N]` files.
    save_ir_model_sharded(
        ir.from_proto(onnx_model), out_dir, os.path.basename(output_onnx_path),
        size_threshold_bytes=INLINE_SIZE_THRESHOLD_BYTES,
        max_shard_size_bytes=MAX_SHARD_SIZE_BYTES,
    )
    print(f"Saved text encoder: {output_onnx_path}")
    return output_onnx_path


def build(input_path, output_dir, precision="f16_int4_quant", extra_options=None):
    if precision not in PRECISION_CONFIGS:
        raise ValueError(f"Unknown precision '{precision}'; choose from {sorted(PRECISION_CONFIGS)}")
    config = PRECISION_CONFIGS[precision]

    # `enable_webgpu_graph=true` selects the GPU-only, Shape-free graph-capture mask reformat;
    # anything else (the default) uses the standard Shape-based form that WebNN EP prefers.
    extra_options = extra_options or {}
    enable_webgpu_graph = str(extra_options.get("enable_webgpu_graph", "false")).lower() not in ("false", "0", "")

    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    suffix = config["suffix"]
    output_onnx = os.path.join(onnx_dir, f"text_encoder_model_{suffix}.onnx")
    export_qwen3_text_encoder(
        input_path, output_onnx, quantize=config["quantize"], dtype=config["dtype"],
        enable_webgpu_graph=enable_webgpu_graph,
    )
    return onnx_dir


def get_args():
    parser = argparse.ArgumentParser(description="Export the Z-Image-Turbo text encoder (Qwen3) to ONNX.")
    parser.add_argument("input_path", help="Path to the Z-Image-Turbo `text_encoder/` folder (or its parent).")
    parser.add_argument(
        "-o", "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the ONNX model to (under an `onnx/` subdir). Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "-p", "--precision", default="f16_int4_quant", choices=sorted(PRECISION_CONFIGS),
        help="Output precision. Default: f16_int4_quant.",
    )
    parser.add_argument(
        "--extra_options", nargs="*", default=[],
        help="Extra key=value options. Supported: enable_webgpu_graph=true|false (default false). "
        "true emits the GPU-only graph-capture attention-mask reformat (WebGPU graph capture); "
        "false emits the standard Shape-based form preferred by WebNN EP.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    input_path = args.input_path
    if os.path.isdir(os.path.join(input_path, "text_encoder")):
        input_path = os.path.join(input_path, "text_encoder")
    onnx_dir = build(input_path, args.output_dir, args.precision, parse_extra_options(args.extra_options))
    print(f"\nSuccess: text_encoder exported to {onnx_dir}")


if __name__ == "__main__":
    main()
