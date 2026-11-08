# awq/_scale.py — AWQ (Lin et al. 2023) per-input-channel scale for linears sharing one input.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# s = mean|x|^alpha (normalised by sqrt(max*min)); quantize W·diag(s), undo with diag(1/s) on
# input side. Salient (high-activation) channels get finer effective steps. alpha picked by grid
# search on real output error ||X Wᵀ − X Q(W s)ᵀ / s|| over a row sample.
import numpy as np
from pipeline.surgery.wquant._rtn_dequant import rtn_dequant


def awq_scale(ws, x, bits, group, grid=20, rows=1024, seed=0):
    rng = np.random.default_rng(seed)
    xs = x[rng.choice(len(x), min(rows, len(x)), replace=False)]
    w = np.concatenate(ws, axis=0)                            # q|k|v or gate|up stacked on output axis
    ref = xs @ w.T
    act = np.abs(x).mean(0) + 1e-8
    best, best_s = None, np.ones_like(act)
    for alpha in np.linspace(0.0, 1.0, grid + 1):
        s = act ** alpha
        s = s / np.sqrt(s.max() * s.min())
        err = np.square(ref - (xs / s) @ rtn_dequant(w * s, bits, group).T).mean()
        if best is None or err < best:
            best, best_s = err, s
    return best_s.astype(np.float32)
