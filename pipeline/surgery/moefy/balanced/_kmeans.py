# balanced/_kmeans.py — equal-size k-means over neurons' activation patterns.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Neurons that fire on the same tokens belong in the same expert (MoEfication, Zhang et
# al. 2022): then a router choosing an expert for a token keeps most of that token's
# active neurons. Feature = unit-normalised |activation| over calibration tokens.
import numpy as np
from pipeline.surgery.moefy.balanced._assign import balanced_assign


def balanced_kmeans(acts, experts, iters=20, seed=0):
    """acts [N, I] -> labels [I] in 0..experts-1, every expert exactly I // experts neurons."""
    f = np.abs(acts).T                                        # [I, N] firing pattern per neuron
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8) # cosine geometry: pattern, not magnitude
    rng = np.random.default_rng(seed)
    cent = f[rng.choice(len(f), experts, replace=False)]      # random distinct neurons as seeds
    labels = None
    for _ in range(iters):
        new = balanced_assign(-(f @ cent.T))                  # distance = negative cosine similarity
        if labels is not None and (new == labels).all():
            break                                             # converged
        labels = new
        cent = np.stack([f[labels == c].mean(0) for c in range(experts)])
    return labels
