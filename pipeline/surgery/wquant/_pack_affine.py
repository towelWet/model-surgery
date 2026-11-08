# _pack_affine.py — pack integer codes [out, in] into MLX uint32 layout (value j of word at bits*j).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Layout verified against mx.quantize output for 4 and 8 bits (little-endian within each word).
import numpy as np


def pack_affine(codes, bits):
    if 32 % bits:
        raise ValueError("pack_affine supports 2, 4, 8 bits (values must not straddle words)")
    per = 32 // bits
    c = codes.astype(np.uint32).reshape(codes.shape[0], -1, per)
    out = np.zeros(c.shape[:2], dtype=np.uint32)
    for j in range(per):
        out |= c[:, :, j] << np.uint32(bits * j)
    return out
