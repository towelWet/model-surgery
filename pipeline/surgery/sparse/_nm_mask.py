# _nm_mask.py — N:M keep-mask: in every M consecutive inputs of a row keep N highest scores.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# 2:4 is format NVIDIA sparse tensor cores accelerate. Apple Metal / MLX has no sparse
# kernel, so on this machine N:M only matters for export elsewhere, not local speed.
import numpy as np


def nm_mask(score, n, m):
    """score [out, in] (in % m == 0) -> bool keep-mask, n of every m kept."""
    out, inp = score.shape
    if inp % m or not 0 < n < m:
        raise ValueError(f"N:M {n}:{m} needs 0 < N < M and input width {inp} divisible by M")
    blocks = score.reshape(out, inp // m, m)
    keep = np.zeros(blocks.shape, dtype=bool)
    top = np.argsort(-blocks, axis=2)[:, :, :n]             # n highest per block
    np.put_along_axis(keep, top, True, axis=2)
    return keep.reshape(out, inp)
