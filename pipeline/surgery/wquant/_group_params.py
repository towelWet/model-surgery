# _group_params.py — MLX affine scale/bias for one column group, taken from mx.quantize itself.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# MLX does not use plain min/max: bias sits at larger-magnitude edge, scale may be negative and
# is snapped so 0.0 is exactly representable. Re-deriving that rule risks drift; asking
# mx.quantize guarantees GPTQ codes dequantize exactly like MLX's own.
import numpy as np
import mlx.core as mx


def group_params(wg, bits):
    """wg [out, group] float32 -> (scale [out], bias [out])."""
    _, s, b = mx.quantize(mx.array(wg, dtype=mx.float32), group_size=wg.shape[1], bits=bits)
    bf = lambda a: np.array(a.astype(mx.bfloat16).astype(mx.float32))[:, 0]  # checkpoint stores bf16: GPTQ must see same grid
    return bf(s), bf(b)
