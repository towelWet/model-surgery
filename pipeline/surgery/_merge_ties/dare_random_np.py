# dare_random_np.py — DARE: Bernoulli keep-mask with inverse-probability rescaling.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np


def dare_random_np(delta, density, rng):
    """DARE preserves each parameter delta in expectation, not each sample's L1 norm; mutates `delta`."""
    if not np.isfinite(density) or not 0 < density <= 1:
        raise ValueError("DARE density must be finite and in (0, 1]")
    if density == 1.0:
        return delta
    # In place on the caller-owned float32 delta: no float64 draw, no masked/rescaled copies.
    # Stacked expert tensors reach ~0.5 G elements, where each extra float64 copy cost ~4 GB.
    # float32 draws resolve density to ~2**-24, far finer than any density a merge uses.
    # Seeded outputs differ from pre-2026-11-07 float64 draws; same distribution, new stream.
    flat = delta.reshape(-1)                                  # view when contiguous (always: fresh delta)
    flat *= rng.random(flat.size, dtype=np.float32) < density  # Bernoulli keep-mask (bool temp: 1 B/elt)
    flat /= np.float32(density)                               # inverse-probability rescale
    return delta
