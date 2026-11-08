# load/all_weights.py — Return (flat weight dict, weight_map or None for single-shard).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import glob
import mlx.core as mx
import os

from .._shared import load_index, load_shard_cache


def load_all_weights(model_dir: str):
    """Return (flat weight dict, weight_map or None for single-shard)."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    cache = {}
    weights = {}

    if os.path.exists(index_path):
        idx = load_index(model_dir)
        wmap = idx["weight_map"]
        for key, shard_file in wmap.items():
            shard = load_shard_cache(model_dir, shard_file, cache)
            weights[key] = shard[key]
        return weights, wmap

    shard_files = sorted(glob.glob(os.path.join(model_dir, "model*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No model*.safetensors in {model_dir}")
    for sf in shard_files:
        weights.update(mx.load(sf))
    return weights, None
