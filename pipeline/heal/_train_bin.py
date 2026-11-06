# _train_bin.py — resolve the tensor-array-train binary (never mlx_lm).
import os
import shutil


def train_bin():
    """TENSOR_ARRAY_TRAIN env first, then PATH lookup; clear error if absent."""
    path = os.environ.get("TENSOR_ARRAY_TRAIN") or shutil.which("tensor-array-train")
    if not path:
        raise FileNotFoundError("tensor-array-train not found (set TENSOR_ARRAY_TRAIN or add to PATH)")
    return path
