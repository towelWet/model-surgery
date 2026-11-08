# _tensors.py — one shard's tensors: affine triples (weight/scales/biases) -> bf16 weight.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Bits / group size per prefix come from config (per-path override wins), same lookup the
# export converter uses. Non-quantized tensors pass through unchanged (norms, A_log, conv1d).
import mlx.core as mx
from pipeline.export._convert_quant.params_for_path import quant_params_for_path


def dequant_tensors(tensors, cfg):
    """tensors: name -> mx array (one shard); cfg: config with "quantization" -> bf16 dict."""
    out = {}
    for k, v in tensors.items():
        if k.endswith((".scales", ".biases")):                 # consumed with their .weight
            continue
        p = k[: -len(".weight")] if k.endswith(".weight") else None
        if p and p + ".scales" in tensors:                     # packed uint32 affine triple
            gs, bits = quant_params_for_path(cfg, p)           # per-tensor override (e.g. 8-bit head)
            v = mx.dequantize(v, tensors[p + ".scales"], tensors[p + ".biases"],
                              group_size=gs, bits=bits).astype(mx.bfloat16)
        out[k] = v
    return out
