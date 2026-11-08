# ../merge/_ties/della_np.py — DELLA MagPrune: magnitude-ranked random drop with per-element rescale.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Deep et al. 2024: keep probability rises linearly with magnitude rank from density-epsilon
# (smallest) to density+epsilon (largest); kept deltas divide by their own probability so
# each is unbiased, like DARE but favouring big changes. Mutates delta.
import numpy as np


def della_np(delta, density, epsilon, rng):
    lo, hi = density - epsilon, density + epsilon
    if not 0 < lo <= hi <= 1:
        raise ValueError("della needs 0 < density - epsilon and density + epsilon <= 1")
    flat = delta.reshape(-1)
    rank = np.empty(flat.size, dtype=np.float32)
    rank[np.argsort(np.abs(flat))] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    p = lo + (hi - lo) * rank                                 # per-element keep probability
    flat *= (rng.random(flat.size, dtype=np.float32) < p) / p
    return delta
