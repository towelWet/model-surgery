# list_shards.py — Return sorted unique safetensor shard basenames for a checkpoint.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import glob
import os

from ._shared import load_index


def list_shard_files(model_dir: str) -> list[str]:
    """Return sorted unique safetensor shard basenames for a checkpoint."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        wmap = load_index(model_dir)["weight_map"]
        return sorted(set(wmap.values()))
    paths = sorted(glob.glob(os.path.join(model_dir, "model*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"No model*.safetensors in {model_dir}")
    return [os.path.basename(p) for p in paths]
