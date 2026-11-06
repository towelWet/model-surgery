# _write.py — write depth-pruned shards; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os

import mlx.core as mx

from shard_io import load_shard_cache, write_index


def write_depth_shards(dest, by_out, shard_cache, src, metadata, new_wmap, eval_every=30):
    for out_shard in sorted(by_out):
        entries = by_out[out_shard]
        print(f"writing {out_shard} ({len(entries)} tensors)...")
        out = {}
        for i, (nk, ok, ss) in enumerate(entries):
            out[nk] = load_shard_cache(src, ss, shard_cache)[ok]
            if (i + 1) % eval_every == 0:
                mx.eval(*out.values())
                print(f"  eval flush @{i + 1}")
        mx.eval(*out.values())
        mx.save_safetensors(os.path.join(dest, out_shard), out)
        print(f"  saved {out_shard}")
    write_index(dest, metadata, new_wmap)
