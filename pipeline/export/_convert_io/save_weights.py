# save_weights.py — Write sharded safetensors + index.json.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx
import os

from ._shared import write_index
from .make_shards import make_shards


def save_weights(dest_dir: str, weights: dict) -> None:
    """Write sharded safetensors + index.json."""
    shards = make_shards(weights)
    n = len(shards)
    fmt = "model-{:05d}-of-{:05d}.safetensors" if n > 1 else "model.safetensors"
    total_size = sum(v.nbytes for v in weights.values())
    wmap = {}

    for i, shard in enumerate(shards):
        name = fmt.format(i + 1, n) if n > 1 else fmt
        path = os.path.join(dest_dir, name)
        mx.save_safetensors(path, shard, metadata={"format": "mlx"})
        for k in shard:
            wmap[k] = name

    meta = {"total_size": total_size}
    write_index(dest_dir, meta, dict(sorted(wmap.items())))
