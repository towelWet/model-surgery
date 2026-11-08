# _load_calib.py — stack one layer's captured MLP inputs from calibration files.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Files come from pipeline/profile/moefy/capture.py, float32 per layer i:
#   "mlp.<i>" [S, H] MLP input; "attn.<i>" [S, H] q/k/v input; "o.<i>" [S, heads*head_dim] o_proj input.
import numpy as np
import mlx.core as mx


def load_calib(files, layer, tap="mlp"):
    """[N, width] float32: every token's input of `tap` (mlp | attn | o) at `layer`, all files."""
    key = f"{tap}.{layer}"
    rows = [np.array(mx.load(f)[key]) for f in files]        # mx.load is lazy; only this key materialises
    return np.concatenate(rows, axis=0)
