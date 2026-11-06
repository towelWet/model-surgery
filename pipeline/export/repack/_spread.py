"""Repack 1-bit weights into 2-bit layout (bit-deposit primitive lives in _bit_deposit.py).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
"""
import numpy as np

from _bit_deposit import spread_bits_16      # re-exported: old importers keep working


def repack_weight(w1: np.ndarray) -> np.ndarray:
    """[N, K/32] uint32 1-bit -> [N, K/16] uint32 2-bit. Vectorized."""
    lo = w1 & 0xFFFF                           # lower 16 bits per word
    hi = (w1 >> 16) & 0xFFFF                   # upper 16 bits per word
    lo_sp = spread_bits_16(lo)                  # spread into 2-bit slots
    hi_sp = spread_bits_16(hi)
    N, p1 = w1.shape
    out = np.empty((N, p1 * 2), dtype=np.uint32)
    out[:, 0::2] = lo_sp                       # interleave lo/hi halves
    out[:, 1::2] = hi_sp
    return out
