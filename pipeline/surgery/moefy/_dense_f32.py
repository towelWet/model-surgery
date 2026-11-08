# _dense_f32.py — one dense weight as float32 numpy, HF [out, in] layout.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
import mlx.core as mx


def dense_f32(w, key):
    return np.array(w[key].astype(mx.float32))               # bf16 has no numpy dtype; widen on device
