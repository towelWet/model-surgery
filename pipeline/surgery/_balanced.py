# _balanced.py — balanced spherical k-means: every cluster gets exactly `size` members.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# MoE experts must be equal width (SwitchGLU stacks them into one tensor), so ordinary
# k-means is unusable. Each round: score every (unit, cluster) pair by cosine similarity,
# then assign greedily from the strongest pair down, skipping clusters already full.
# Units are rows of `feats` (activation profiles across calibration tokens).
import numpy as np


def balanced_kmeans(feats, k, size, iters=15, seed=0):
    """Return labels[n] in [0,k) with exactly `size` units per label (n == k*size)."""
    n = feats.shape[0]
    if n != k * size:
        raise ValueError(f"{n} units cannot split into {k} x {size}")
    x = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)   # cosine geometry
    rng = np.random.default_rng(seed)
    cent = x[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        sim = x @ cent.T                                                  # [n, k]
        order = np.argsort(-sim, axis=None)                               # strongest pairs first
        labels[:] = -1
        fill = np.zeros(k, dtype=np.int64)
        for flat in order:
            u, c = divmod(int(flat), k)
            if labels[u] < 0 and fill[c] < size:
                labels[u] = c
                fill[c] += 1
                if fill.sum() == n:
                    break
        new = np.stack([x[labels == c].mean(0) for c in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-8
        if np.allclose(new, cent, atol=1e-5):
            break
        cent = new
    return labels
