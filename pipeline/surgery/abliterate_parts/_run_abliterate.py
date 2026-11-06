# _run_abliterate.py — refusal-direction projection on weights; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx

from projection.apply import project_out
from _data import TARGET_ENDS
from _pack_shards import save_weight_dict
from shard_io import load_index, load_shard_cache, copy_tokenizer


def run_abliterate(src, dest, r_hat, alpha, cfg):
    idx = load_index(src)
    wmap = idx["weight_map"]
    shard_cache = {}
    out = {}
    EVAL_EVERY = 15


    copy_tokenizer(src, dest)
    import json
    import os

    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for old_key, shard in sorted(wmap.items()):
        t = load_shard_cache(src, shard, shard_cache)[old_key]
        if any(old_key.endswith(e) for e in TARGET_ENDS):
            t = project_out(t, r_hat, alpha)
        out[old_key] = t
        if len(out) % EVAL_EVERY == 0:
            mx.eval(*out.values())
    mx.eval(*out.values())
    save_weight_dict(dest, out, wmap, idx.get("metadata", {}))
