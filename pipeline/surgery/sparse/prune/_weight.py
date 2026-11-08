# prune/_weight.py — zero lowest-scoring weights of one bf16 matrix; return it and its zero fraction.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# score = |W| (magnitude) or |W| * ||x_j|| (Wanda, when input-channel norms are given).
# Kept weights are the ORIGINAL bf16 values (mx.where), never a float32 round trip.
import numpy as np
import mlx.core as mx
from pipeline.surgery.sparse._row_mask import row_mask
from pipeline.surgery.sparse._nm_mask import nm_mask


def prune_weight(w, sparsity, norms=None, nm=None):
    """w mx [out, in]; nm = (n, m) or None. Returns (pruned mx array, zero fraction)."""
    score = np.abs(np.array(w.astype(mx.float32)))
    if norms is not None:
        score = score * norms[None, :]                     # Wanda: weight x input activation size
    keep = nm_mask(score, *nm) if nm else row_mask(score, sparsity)
    out = mx.where(mx.array(keep), w, mx.zeros_like(w))
    mx.eval(out)                                           # per tensor: whole-shard graph at save hit GPU watchdog
    return out, float(1.0 - keep.mean())
