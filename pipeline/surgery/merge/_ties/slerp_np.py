# ../merge/_ties/slerp_np.py — spherical interpolation between two tensors (mergekit semantics).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Angle comes from unit-normalised copies; interpolation is applied to ORIGINAL tensors.
# Nearly parallel tensors (|cos| > 0.9995) fall back to linear interpolation: sin(theta)
# is ~0 there and spherical weights blow up numerically.
import numpy as np


def slerp_np(a, b, t):
    """t = 0 -> a, t = 1 -> b, same shape float32."""
    a32, b32 = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
    na, nb = np.linalg.norm(a32), np.linalg.norm(b32)
    dot = float(np.dot(a32, b32) / (na * nb)) if na > 0 and nb > 0 else 1.0
    if abs(dot) > 0.9995:
        out = (1 - t) * a32 + t * b32
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        s = np.sin(theta)
        out = (np.sin((1 - t) * theta) / s) * a32 + (np.sin(t * theta) / s) * b32
    return out.reshape(a.shape).astype(np.float32)
