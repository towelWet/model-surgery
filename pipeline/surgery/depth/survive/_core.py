# survive/_core.py — surviving tensor keys after layer removal; logic only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.surgery._keys import remap_key
from pipeline.surgery._layer_prefix import decoder_layer_prefix
from pipeline.surgery.depth.survive._map import shard_map_for_survivors   # re-export for depth/prune.py


def surviving_triples(wmap, shard_order, remove_set, layers=None):
    prefix = decoder_layer_prefix(wmap, layers)          # layers = source decoder count (VLM tie-break)
    return [
        (remap_key(ok, remove_set, prefix), ok, s)
        for s in shard_order
        for ok, sv in wmap.items()
        if sv == s
        for _ in [None]
        if remap_key(ok, remove_set, prefix) is not None
    ]
