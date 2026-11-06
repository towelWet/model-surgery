# requant_mx.py — Quantize host fp32 numpy back to materialized MLX affine triple.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx

from ._shared import BITS, GROUP_SIZE
from .materialize import materialize


def requant_mx(w_np, gs=GROUP_SIZE, bits=BITS):
    """Quantize host fp32 numpy back to materialized MLX affine triple."""
    w_q, scales, biases = mx.quantize(
        mx.array(w_np, dtype=mx.bfloat16), group_size=gs, bits=bits
    )
    return materialize(w_q), materialize(scales), materialize(biases)
