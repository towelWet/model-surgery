# _quant_linear.py — one float weight -> MLX packed (weight, scales, biases) by RTN or GPTQ.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# AWQ uses RTN here: its gain comes from scales folded in beforehand (_awq_fold).
import numpy as np
import mlx.core as mx
from pipeline.surgery.wquant._gptq_matrix import gptq_matrix
from pipeline.surgery.wquant._pack_affine import pack_affine


def quant_linear(method, w, x, bits, group):
    if method != "gptq":
        return mx.quantize(mx.array(w, dtype=mx.float32), group_size=group, bits=bits)
    h = 2.0 * (x.T.astype(np.float64) @ x) / len(x)
    codes, s, b = gptq_matrix(w, h, bits, group)
    return mx.array(pack_affine(codes, bits)), mx.array(s), mx.array(b)
