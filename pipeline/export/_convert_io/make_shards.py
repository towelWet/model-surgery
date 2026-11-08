# make_shards.py — Split weight dict into shard-sized chunks (ported from mlx_lm utils.make_shards).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from ._shared import MAX_FILE_SIZE_GB


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """Split weight dict into shard-sized chunks (ported from mlx_lm utils.make_shards)."""
    max_bytes = max_file_size_gb << 30
    shards, shard, shard_size = [], {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_bytes and shard:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    if shard:
        shards.append(shard)
    return shards
