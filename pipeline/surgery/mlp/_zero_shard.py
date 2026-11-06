# _zero_shard.py — zero MLP weights for selected layers in one shard; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx

from shard_io import load_shard_cache
from _mlp_layer import mlp_layer                     # prefix-aware (VLM-nested keys too)


def process_shard(src, shard, keys, zero_layers, shard_cache, eval_every=30):
    data = load_shard_cache(src, shard, shard_cache)
    out = {}
    for i, ok in enumerate(keys):
        t = data[ok]
        out[ok] = mx.zeros_like(t) if mlp_layer(ok) in zero_layers else t   # None never in set
        if (i + 1) % eval_every == 0:
            mx.eval(*out.values())                   # bound lazy graph / RAM
    mx.eval(*out.values())
    return out
