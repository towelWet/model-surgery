# _svd_factor.py — rank-r factors A [out, r], B [r, in] with W ≈ A @ B (plain SVD or ASVD).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# ASVD (Yuan et al. 2023): factor W·diag(s) where s = input-channel activation norm ** alpha,
# then undo scale on B. Truncation error then lands on channels real inputs barely use,
# instead of being spread evenly as plain SVD does. sqrt(S) split balances A and B magnitudes.
import numpy as np


def svd_factor(w, rank, scale=None):
    w = np.asarray(w, dtype=np.float32)
    s = np.ones(w.shape[1], dtype=np.float32) if scale is None else np.maximum(scale, 1e-6).astype(np.float32)
    u, sv, vt = np.linalg.svd(w * s, full_matrices=False)  # scale columns = input channels
    root = np.sqrt(sv[:rank])
    a = u[:, :rank] * root                                   # [out, r]
    b = (root[:, None] * vt[:rank]) / s                      # [r, in], scale removed again
    return a, b
