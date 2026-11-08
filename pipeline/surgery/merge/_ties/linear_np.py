# ../merge/_ties/linear_np.py — Model Soup: weighted average of whole fine-tuned weights (not task vectors).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Wortsman et al. 2022. Base is not averaged in; list base again as a source to include it.
import numpy as np


def linear_np(others, weights):
    """sum(w_i * theta_i) / sum(w_i); weights must not sum to zero."""
    total = float(sum(weights))
    if total == 0.0:
        raise ValueError("linear merge weights sum to zero")
    out = np.zeros(others[0].shape, dtype=np.float32)
    for other, w in zip(others, weights):
        out += np.float32(w) * other.astype(np.float32)
    return out / np.float32(total)
