import mlx.core as mx
import math


def project_out(weight, direction, alpha):
    weight = weight.astype(mx.float32)
    direction = mx.array(direction, dtype=mx.float32)
    if weight.ndim != 2 or direction.shape != (weight.shape[0],):
        raise ValueError("output-space direction must match matrix rows")
    norm = float(mx.linalg.norm(direction).item())
    if not math.isfinite(alpha) or not math.isfinite(norm) or norm == 0:
        raise ValueError("alpha and nonzero direction must be finite")
    direction = direction / norm
    return weight - alpha * mx.outer(direction, direction @ weight)
