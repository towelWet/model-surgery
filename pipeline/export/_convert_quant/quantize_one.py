# quantize_one.py — affine-quantize one weight (via bf16) to (packed, scales, biases).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx

from ._shared import materialize


def quantize_one(
    weight: mx.array,
    group_size: int,
    bits: int,
) -> tuple[mx.array, mx.array, mx.array]:
    w_q, scales, biases = mx.quantize(
        weight.astype(mx.bfloat16),
        group_size=group_size,
        bits=bits,
    )
    return materialize(w_q), materialize(scales), materialize(biases)
