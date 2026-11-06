# _survive_map.py — assign surviving tensors to evenly filled output shards.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _shard_name import shard_name


def shard_map_for_survivors(surviving, n_shards):
    chunk = (len(surviving) + n_shards - 1) // n_shards
    return {nk: shard_name(i // chunk + 1, n_shards) for i, (nk, _, _) in enumerate(surviving)}
