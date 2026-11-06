# _regroup_shards.py — spread keys evenly over same number of output shards: {out_shard: [(key, src_shard)]}.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _shard_name import shard_name


def regroup_shards(wmap):
    n_shards = len(set(wmap.values()))
    keys = list(wmap)
    chunk = (len(keys) + n_shards - 1) // n_shards          # ceil: every key lands somewhere
    new_wmap = {k: shard_name(i // chunk + 1, n_shards) for i, k in enumerate(keys)}
    by_out = {}
    for k in keys:
        by_out.setdefault(new_wmap[k], []).append((k, wmap[k]))  # (key, src_shard)
    return new_wmap, by_out
