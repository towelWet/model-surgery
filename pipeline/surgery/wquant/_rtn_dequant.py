# _rtn_dequant.py — round-to-nearest MLX affine quantize then dequantize (float result, same shape).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
import mlx.core as mx


def rtn_dequant(w, bits, group):
    q, s, b = mx.quantize(mx.array(w, dtype=mx.float32), group_size=group, bits=bits)
    return np.array(mx.dequantize(q, s, b, group_size=group, bits=bits))
