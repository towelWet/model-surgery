# merge_delta_np.py — Task vector from other − base, DARE prune, add back (numpy, host-side).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from .dare_random_np import dare_random_np


def merge_delta_np(base_w, other_w, density, weight_b, lam, rng):
    """Task vector from other − base, DARE prune, add back (numpy, host-side)."""
    delta = (other_w - base_w) * weight_b
    delta = dare_random_np(delta, density, rng)
    return base_w + lam * delta
