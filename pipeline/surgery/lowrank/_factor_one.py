# _factor_one.py — factor one weight: dense reconstruction, bf16 factors, report row.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
import mlx.core as mx
from pipeline.surgery.lowrank._svd_factor import svd_factor
from pipeline.surgery.lowrank._keep_rank import keep_rank


def factor_one(key, weight, ratio, scale):
    """scale: ASVD input-channel weights (already ** alpha) or None for plain SVD."""
    w = np.array(weight.astype(mx.float32))
    r = keep_rank(*w.shape, ratio)
    a, b = svd_factor(w, r, scale)
    dense = mx.array(a @ b).astype(weight.dtype)
    err = float(np.linalg.norm(w - a @ b) / np.linalg.norm(w))
    return dense, mx.array(a).astype(mx.bfloat16), mx.array(b).astype(mx.bfloat16), {"key": key, "rank": r, "rel_err": err}
