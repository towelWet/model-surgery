# dequant_np.py — Dequant to host fp32 numpy — keeps merge math off the Metal graph.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx

from ._shared import BITS, GROUP_SIZE
from .mlx_to_np_f32 import mlx_to_np_f32


def dequant_np(shard, prefix, gs=GROUP_SIZE, bits=BITS):
    """Dequant to host fp32 numpy — keeps merge math off the Metal graph."""
    w = mx.dequantize(
        shard[prefix + ".weight"],
        shard[prefix + ".scales"],
        shard[prefix + ".biases"],
        group_size=gs,
        bits=bits,
    ).astype(mx.float32)
    mx.eval(w)
    return mlx_to_np_f32(w)
