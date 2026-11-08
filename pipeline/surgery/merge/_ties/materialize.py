# ../merge/_ties/materialize.py — Force eval and detach from MLX graph to prevent GPU timeout.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx
import numpy as np


def materialize(t):
    """Force eval and detach from MLX graph to prevent GPU timeout."""
    mx.eval(t)
    if t.dtype == mx.bfloat16:
        arr = np.array(t.astype(mx.float32), dtype=np.float32)
        return mx.array(arr).astype(mx.bfloat16)
    return mx.array(np.array(t))
