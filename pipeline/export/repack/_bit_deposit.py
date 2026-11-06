# _bit_deposit.py — spread 16 source bits into 32 (each bit gets its own 2-bit slot).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np


def spread_bits_16(x: np.ndarray) -> np.ndarray:
    """Spread 16 lower bits of each uint32 into 32 bits (bit -> 2-bit slot).
    Classic parallel bit-deposit: interleave zeros between each source bit."""
    x = x.astype(np.uint64)                    # room for intermediate shifts
    x = (x | (x << 8)) & np.uint64(0x00FF00FF)  # groups of 8 -> spaced
    x = (x | (x << 4)) & np.uint64(0x0F0F0F0F)  # groups of 4
    x = (x | (x << 2)) & np.uint64(0x33333333)  # groups of 2
    x = (x | (x << 1)) & np.uint64(0x55555555)  # individual bits
    return x.astype(np.uint32)
