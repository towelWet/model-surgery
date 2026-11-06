# dare_random_np.py — DARE: Bernoulli keep-mask at density, L1-rescale to preserve magnitude.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np


def dare_random_np(delta, density, rng):
    """DARE: Bernoulli keep-mask at density, L1-rescale to preserve magnitude."""
    if density >= 1.0:
        return delta
    flat = delta.reshape(-1)
    mask = rng.random(flat.size) < density
    masked = flat * mask
    before = np.abs(flat).sum()
    after = np.abs(masked).sum()
    if after > 1e-9 and before > 1e-9:
        masked *= before / after
    return masked.reshape(delta.shape)
