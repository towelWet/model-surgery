# width/prune.py — slice dense MLP gate/up/down_proj to a smaller intermediate_size (no dequant).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 width/prune.py <src_dir> <dest_dir> <new_intermediate>
# Keeps the FIRST new_intermediate neurons. That is only meaningful if neurons were already
# reordered by importance (profile/mlp/stats.py); on a raw checkpoint it is an arbitrary cut.
# Group alignment: new_intermediate must be divisible by quant group_size (packed groups can't split).
import json
import os
import sys

import mlx.core as mx
mx.set_default_device(mx.cpu)                                  # host slicing; GPU save hit watchdog while server ran
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.width._regroup_shards import regroup_shards
from pipeline.surgery.width._slice_mlp import slice_mlp
from pipeline.surgery.shard_io import copy_tokenizer, load_index, load_shard_cache, write_index
from pipeline.surgery.checkpoint.destination import validate_destination

src, dest, new_interm = sys.argv[1], sys.argv[2], int(sys.argv[3])
validate_destination(dest, sources=[src])                  # Tau name + overlap before any writes
idx = load_index(src)
with open(os.path.join(src, "config.json")) as f:
    cfg = json.load(f)
tc = cfg.get("text_config", cfg)                           # VLM nests text config
old_interm = tc["intermediate_size"]
gsize = cfg.get("quantization", {}).get("group_size", 1)   # bf16 checkpoint: no group constraint
if not 0 < new_interm < old_interm:
    raise ValueError(f"new {new_interm} must satisfy 0 < new < current {old_interm}")
if new_interm % gsize != 0:
    raise ValueError(f"new_intermediate {new_interm} not divisible by group_size {gsize}")

os.makedirs(dest)
copy_tokenizer(src, dest)
tc["intermediate_size"] = new_interm
with open(os.path.join(dest, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

new_wmap, by_out = regroup_shards(idx["weight_map"])
shard_cache = {}
for out_shard in sorted(by_out):
    print(f"writing {out_shard} ({len(by_out[out_shard])} tensors)...")
    out = {k: slice_mlp(k, load_shard_cache(src, ss, shard_cache)[k], old_interm, new_interm)
           for k, ss in by_out[out_shard]}
    mx.save_safetensors(os.path.join(dest, out_shard), out)
write_index(dest, idx.get("metadata", {}), new_wmap)
print(f"Done -> {dest}  intermediate_size: {old_interm} -> {new_interm}")  # old printed new->new
