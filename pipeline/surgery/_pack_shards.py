# _pack_shards.py — write flat weight dict into safetensor shards; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os

import mlx.core as mx

from _keys import shard_name


def save_weight_dict(dest, out, wmap, metadata=None):
    """Pack `out` into same shard count as source weight_map."""
    all_keys = sorted(out.keys())
    n_shards = len(set(wmap.values()))
    chunk = (len(all_keys) + n_shards - 1) // n_shards
    new_wmap = {k: shard_name(i // chunk + 1, n_shards) for i, k in enumerate(all_keys)}
    by_out = {}
    for k in all_keys:
        by_out.setdefault(new_wmap[k], []).append(k)
    for out_shard in sorted(by_out):
        keys = by_out[out_shard]
        print(f"writing {out_shard} ({len(keys)} tensors)...")
        mx.save_safetensors(os.path.join(dest, out_shard), {k: out[k] for k in keys})
    from shard_io import write_index

    write_index(dest, metadata or {}, new_wmap)
