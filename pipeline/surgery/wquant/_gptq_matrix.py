# _gptq_matrix.py — GPTQ (Frantar et al. 2022) for one linear: quantize columns left to right,
# pushing each column's rounding error onto not-yet-quantized columns via inverse Hessian.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# H = 2 XᵀX / N from captured inputs. Damping 1 % of mean diagonal keeps Cholesky stable.
# Group params come from current (error-updated) weights when a group starts, as in reference
# code; no act-order (column permutation would break contiguous MLX groups).
import numpy as np
from pipeline.surgery.wquant._group_params import group_params


def gptq_matrix(w, h, bits, group, block=128, damp=0.01):
    w = w.astype(np.float64).copy()
    h = h.astype(np.float64).copy()
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0                                          # input never active: weight irrelevant
    h[np.diag_indices_from(h)] += damp * np.mean(np.diag(h))
    hinv = np.linalg.cholesky(np.linalg.inv(h)).T             # upper Cholesky of H⁻¹
    n_out, n_in = w.shape
    codes = np.zeros((n_out, n_in), dtype=np.uint32)
    scales = np.zeros((n_out, n_in // group), dtype=np.float32)
    biases = np.zeros_like(scales)
    maxq = 2 ** bits - 1
    for i1 in range(0, n_in, block):
        i2 = min(i1 + block, n_in)
        w1, err1, hi = w[:, i1:i2].copy(), np.zeros((n_out, i2 - i1)), hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:                              # new group: params from current weights
                g = col // group
                scales[:, g], biases[:, g] = group_params(w1[:, i:i + group].astype(np.float32), bits)
            s, b = scales[:, col // group], biases[:, col // group]
            q = np.clip(np.rint((w1[:, i] - b) / s), 0, maxq)
            codes[:, col] = q
            e = (w1[:, i] - (q * s + b)) / hi[i, i]
            w1[:, i:] -= np.outer(e, hi[i, i:])               # error feedback inside block
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]                  # lazy batch update of later blocks
    return codes, scales, biases
