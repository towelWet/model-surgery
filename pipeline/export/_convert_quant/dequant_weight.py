# dequant_weight.py — Dequantize an affine triple to bf16 for re-quant.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx

from ._shared import materialize
from .params_for_path import quant_params_for_path


def dequant_weight(
    weights: dict,
    prefix: str,
    config: dict,
) -> mx.array:
    """Dequantize an affine triple to bf16 for re-quant."""
    gs, bits = quant_params_for_path(config, prefix)
    w = mx.dequantize(
        weights[prefix + ".weight"],
        weights[prefix + ".scales"],
        weights[prefix + ".biases"],
        group_size=gs,
        bits=bits,
    )
    return materialize(w.astype(mx.bfloat16))
