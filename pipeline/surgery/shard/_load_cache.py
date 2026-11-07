# _load_cache.py — cached shard loader for safetensors files.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import mlx.core as mx


def load_shard_cache(model_dir, shard_filename, cache):
    """Load shard into cache if not already present; return weight dict."""
    if shard_filename not in cache:
        cache.clear()                                    # retain at most one source shard, not the full model
        cache[shard_filename] = mx.load(
            os.path.join(model_dir, shard_filename)
        )                                                # mx.load -> {key: mx.array}
    return cache[shard_filename]
