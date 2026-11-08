# ../merge/_ties/magnitude_trim_np.py — TIES trim: keep largest-|delta| fraction of a task vector, zero rest.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np


def magnitude_trim_np(delta, density):
    """Mutates `delta`; no rescale (TIES paper / mergekit `magnitude` default)."""
    if not np.isfinite(density) or not 0 < density <= 1:
        raise ValueError("TIES density must be finite and in (0, 1]")
    if density == 1.0:
        return delta
    flat = delta.reshape(-1)                                  # view: fresh contiguous delta
    k = int(density * flat.size)                              # same rounding as mergekit
    if k == 0:
        raise ValueError("density keeps zero elements of this tensor")
    mags = np.abs(flat)                                       # one float32 temp
    cut = np.argpartition(mags, flat.size - k)[: flat.size - k]  # indices of smallest n-k magnitudes
    flat[cut] = 0                                             # exactly k survivors (ties broken by partition)
    return delta
