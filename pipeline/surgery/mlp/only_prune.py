# only_prune.py — zero MLP for selected layers; keep attention.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 only_prune.py <src> <dest-Tau-...> "<spec>"
import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.mlp._zero_shard import process_shard
from pipeline.surgery.mlp._tag_config import tag_config
from pipeline.surgery._spec import format_spec, parse_spec
from pipeline.surgery.shard_io import copy_tokenizer, load_index, write_index
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery._layer_prefix import decoder_layer_prefix

if len(sys.argv) < 4:
    print('Usage: only_prune.py <src> <dest-Tau-...> "<spec>"')
    sys.exit(1)
src, dest, spec = sys.argv[1], sys.argv[2], sys.argv[3]
validate_destination(dest, sources=[src])                 # Tau name + overlap before any writes

zero_layers = parse_spec(spec)
idx = load_index(src)
wmap = idx["weight_map"]
with open(os.path.join(src, "config.json")) as f:
    cfg = json.load(f)                                    # read-only; tag_config writes dest config
prefix = decoder_layer_prefix(wmap, cfg.get("text_config", cfg).get("num_hidden_layers"))  # VLM/mtp tie-break
os.makedirs(dest)
shard_order = sorted(set(wmap.values()))
shard_cache = {}
by_shard = {}
for ok, shard in wmap.items():
    by_shard.setdefault(shard, []).append(ok)

copy_tokenizer(src, dest)
tag_config(src, dest, zero_layers)                   # config.json + mlp_zeroed_layers

for shard in shard_order:
    keys = by_shard.get(shard, [])
    print(f"shard {shard} ({len(keys)} tensors)...")
    out = process_shard(src, shard, keys, zero_layers, shard_cache, prefix)
    mx.save_safetensors(os.path.join(dest, shard), out)

write_index(dest, idx.get("metadata", {}), dict(wmap))
print(f"Done -> {dest}  MLP zeroed layers: {format_spec(zero_layers)}")
