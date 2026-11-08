# ../merge/_ties/nuslerp_np.py — NuSLERP: row-wise spherical interpolation of two task vectors, added to base.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# mergekit "nuslerp" idea: interpolate per output row (each neuron's weights) instead of whole
# flattened tensor, and on deltas from base so shared base is not bent. t = w_b / (w_a + w_b).
import numpy as np
from .slerp_np import slerp_np


def nuslerp_np(base_w, a, b, weights):
    if len(weights) != 2 or sum(weights) == 0:
        raise ValueError("nuslerp needs exactly two fine-tunes with nonzero total weight")
    t = float(weights[1]) / float(sum(weights))
    base = base_w.astype(np.float32)
    da, db = a.astype(np.float32) - base, b.astype(np.float32) - base
    if da.ndim < 2:
        return base + slerp_np(da, db, t)                    # vectors (norms, biases): whole slerp
    rows = [slerp_np(ra, rb, t) for ra, rb in zip(da.reshape(da.shape[0], -1), db.reshape(db.shape[0], -1))]
    return base + np.stack(rows).reshape(da.shape)
