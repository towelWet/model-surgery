# ../merge/_ties/mlx_to_np_f32.py — Host float32 numpy from MLX; avoids PEP 3118 bfloat16 breakage on Py3.14.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx
import numpy as np


def mlx_to_np_f32(t):
    """Host float32 numpy from MLX; avoids PEP 3118 bfloat16 breakage on Py3.14."""
    mx.eval(t)
    return np.array(t.astype(mx.float32), dtype=np.float32)
