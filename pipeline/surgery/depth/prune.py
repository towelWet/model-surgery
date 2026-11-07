# prune.py — remove arbitrary layers from a 3-bit checkpoint.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 depth_prune.py <src_dir> <dest_dir> "<spec>"
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline.surgery.depth.validation import validate_depth
from pipeline.surgery.checkpoint.destination import validate_destination

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _survive import shard_map_for_survivors, surviving_triples
from _write import write_depth_shards
from _spec import format_spec, parse_spec
from shard_io import copy_tokenizer, load_index

src, dest = sys.argv[1], sys.argv[2]
validate_destination(dest, sources=[src])                 # Tau name + overlap before any writes
if len(sys.argv) >= 5:
    remove_set = set(range(int(sys.argv[3]), int(sys.argv[4]) + 1))
else:
    remove_set = parse_spec(sys.argv[3])

with open(os.path.join(src, "config.json")) as f:
    cfg = validate_depth(json.load(f), remove_set, dest)

idx = load_index(src)
wmap = idx["weight_map"]
shard_order = sorted(set(wmap.values()))
surviving = surviving_triples(wmap, shard_order, remove_set, cfg.get("text_config", cfg)["num_hidden_layers"] + len(remove_set))  # pre-removal count
new_wmap = shard_map_for_survivors(surviving, len(shard_order))

os.makedirs(dest)
copy_tokenizer(src, dest)
tc = cfg.get("text_config", cfg)
new_n = tc["num_hidden_layers"]
with open(os.path.join(dest, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print(f"config: num_hidden_layers={new_n}")

by_out = {}
for nk, ok, ss in surviving:
    by_out.setdefault(new_wmap[nk], []).append((nk, ok, ss))

shard_cache = {}
write_depth_shards(dest, by_out, shard_cache, src, idx.get("metadata", {}), new_wmap)
print(f"Done -> {dest}  layers: removed {format_spec(remove_set)}  remaining: {new_n}")
