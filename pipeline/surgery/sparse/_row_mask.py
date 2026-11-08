# _row_mask.py — keep-mask zeroing lowest-scoring weights in every output row.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Per-row (per output neuron) comparison is what Wanda found works for LLMs: a global or
# per-matrix threshold empties some rows entirely and kills those outputs.
import numpy as np


def row_mask(score, sparsity):
    """score [out, in] -> bool keep-mask with exactly floor(in * sparsity) zeros per row."""
    drop = int(score.shape[1] * sparsity)
    keep = np.ones(score.shape, dtype=bool)
    if drop:
        idx = np.argpartition(score, drop - 1, axis=1)[:, :drop]  # lowest `drop` scores per row
        np.put_along_axis(keep, idx, False, axis=1)
    return keep
