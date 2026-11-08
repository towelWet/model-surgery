# ../merge/_ties/delta_np.py — Task vector from other − base, DARE prune, add back (numpy, host-side).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from .dare_random_np import dare_random_np
import numpy as np


def merge_delta_np(base_w, other_w, density, weight_b, lam, rng):
    """Task vector from other − base, DARE prune, add back (numpy, host-side)."""
    if base_w.shape != other_w.shape:
        raise ValueError("merge tensor shapes must match exactly")
    if not np.isfinite(weight_b) or not np.isfinite(lam):
        raise ValueError("merge weights must be finite")
    delta = np.subtract(other_w, base_w, dtype=np.float32)   # one fresh float32 buffer reused below
    delta *= np.float32(weight_b)
    delta = dare_random_np(delta, density, rng)             # in place
    delta *= np.float32(lam)
    delta += base_w                                         # merged weight lands in the same buffer
    return delta
