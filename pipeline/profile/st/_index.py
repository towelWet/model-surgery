# st/_index.py — map every tensor name to its shard + byte range across a sharded model.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json, os

from pipeline.profile.st._header import header


def index(model_dir):
    """Map tensor name -> (shard_path, entry, data_start) across all shards."""
    wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    out, cache = {}, {}
    for name, shard in wm.items():
        p = os.path.join(model_dir, shard)
        if p not in cache:
            cache[p] = header(p)                             # parse each shard header once
        table, start = cache[p]
        out[name] = (p, table[name], start)
    return out
