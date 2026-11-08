# balanced/_assign.py — assign items to clusters by nearest distance with equal cluster sizes.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Stacked expert tensors need every expert the same width, so plain k-means (uneven
# clusters) cannot be used. Greedy: walk all (item, cluster) pairs nearest-first and
# take each pair while the item is free and the cluster has room.
import numpy as np


def balanced_assign(dist):
    """dist [I, E] -> labels [I], each cluster gets exactly I // E items (I % E must be 0)."""
    n, e = dist.shape
    if n % e:
        raise ValueError(f"{n} neurons do not split evenly into {e} experts")
    cap = n // e
    labels = np.full(n, -1)
    fill = np.zeros(e, dtype=int)
    for flat in np.argsort(dist, axis=None):                  # nearest pairs first
        i, c = divmod(int(flat), e)
        if labels[i] < 0 and fill[c] < cap:
            labels[i] = c
            fill[c] += 1
    return labels
