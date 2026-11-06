# _prune_core.py — stream every tensor through expert_tensor and save; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os

import mlx.core as mx

from _expert_tensor import expert_tensor
from _pack_shards import save_weight_dict
from shard_io import load_index, load_shard_cache

EVAL_EVERY = 20                                             # materialize often: bounds lazy-graph memory


def prune_experts(src, dest, plan, cfg, n_experts):
    idx = load_index(src)
    wmap = idx["weight_map"]
    shard_cache, out = {}, {}
    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)                         # caller already set new num_experts
    for old_key, shard in sorted(wmap.items()):
        t = load_shard_cache(src, shard, shard_cache)[old_key]
        for key, tensor in expert_tensor(old_key, t, plan, n_experts):
            out[key] = tensor
            if len(out) % EVAL_EVERY == 0:
                mx.eval(*out.values())
    mx.eval(*out.values())
    save_weight_dict(dest, out, wmap, idx.get("metadata", {}))
