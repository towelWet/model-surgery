# quantize_prefix.py — decide + apply quantization for one .weight prefix; returns (tensors, cfg entry).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from ._shared import materialize
from .dequant_weight import dequant_weight
from .is_quantized_prefix import is_quantized_prefix
from .quantize_one import quantize_one
from .should_quantize_weight import should_quantize_weight


def quantize_prefix(weights, weight_map, config, predicate, prefix, default_group_size, default_bits):
    w = weights[prefix + ".weight"]
    if is_quantized_prefix(prefix, weight_map or weights):
        w = dequant_weight(weights, prefix, config)         # re-quant from bf16, never quant-of-quant
    if not should_quantize_weight(prefix, w, default_group_size):
        return {prefix + ".weight": materialize(w)}, None   # norms / odd shapes stay float
    params = predicate(prefix)                              # recipe: False | True | {bits, group_size, mode}
    if params is False:
        return {prefix + ".weight": materialize(w)}, None
    entry = None
    if isinstance(params, dict):
        gs = int(params.get("group_size", default_group_size))
        bits = int(params["bits"])
        entry = {"group_size": gs, "bits": bits, "mode": params.get("mode", "affine")}  # per-prefix override
    else:
        gs, bits = default_group_size, default_bits
    w_q, scales, biases = quantize_one(w, gs, bits)
    return {prefix + ".weight": w_q, prefix + ".scales": scales, prefix + ".biases": biases}, entry
