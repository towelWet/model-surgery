"""Validate and patch output columns in a dequantized [output, input] matrix."""
import math
import mlx.core as mx


def patch_columns(weight, spec):
    direction = mx.array(spec["direction"], dtype=mx.float32)
    columns, alpha = spec["columns"], spec["alpha"]
    if weight.ndim != 2 or direction.shape != (weight.shape[0],):
        raise ValueError("direction must match the weight's output dimension")
    if not bool(mx.all(mx.isfinite(direction)).item()) or not math.isfinite(alpha):
        raise ValueError("direction and alpha must be finite")
    if not columns or any(type(c) is not int or not 0 <= c < weight.shape[1] for c in columns):
        raise ValueError("columns must be nonempty integer indices within the input dimension")
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate column indices are forbidden")
    for col in columns:
        weight = weight.at[:, col].add(alpha * direction)
    return weight
