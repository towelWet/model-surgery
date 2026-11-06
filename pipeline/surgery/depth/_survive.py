# _survive.py — surviving tensor keys after layer removal; logic only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _keys import remap_key
from _survive_map import shard_map_for_survivors   # re-export for depth/prune.py


def surviving_triples(wmap, shard_order, remove_set):
    return [
        (remap_key(ok, remove_set), ok, s)
        for s in shard_order
        for ok, sv in wmap.items()
        if sv == s
        for _ in [None]
        if remap_key(ok, remove_set) is not None
    ]
