# _save_moefy.py — flush and write moefy output shards; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # surgery/ parent
from _keys import shard_name                             # shard filename helper
from shard_io import write_index                         # index writer


def save_moefy(dest, out, wmap, idx_meta, K, S, H, W):
    mx.eval(*out.values())
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
    write_index(dest, idx_meta, new_wmap)
    active = K * S + H
    print(f"\nDone -> {dest}")
    print(f"Active intermediate/token: {active} ({K}×{S}+{H}) vs {W} = {active/W:.1%}")
