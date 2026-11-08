# _down_proj.py — load + dequantize one layer's down_proj from sharded safetensors.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import mlx.core as mx


def down_proj(model, wm, i, GS=64, BITS=3):
    """Dequantize layer i down_proj → float32 [5120, 17408]."""
    p = f"language_model.model.layers.{i}.mlp.down_proj"
    shard = mx.load(os.path.join(model, wm[p + ".weight"]))                 # lazy: only touched tensors load
    return mx.dequantize(
        shard[p + ".weight"],
        shard[p + ".scales"],
        shard[p + ".biases"],
        group_size=GS,
        bits=BITS,
    ).astype(mx.float32)                                                     # [5120, 17408]
