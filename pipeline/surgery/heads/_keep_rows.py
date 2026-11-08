# _keep_rows.py — row (or column) indices kept for surviving heads, head_dim rows per head.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np


def keep_rows(heads, head_dim):
    """Sorted head ids -> flat indices [h*D .. h*D+D-1] for each, in head order."""
    return np.concatenate([np.arange(h * head_dim, (h + 1) * head_dim) for h in heads])
