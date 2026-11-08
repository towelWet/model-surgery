"""Write a tiny genuine MLX checkpoint; split companion tensors across shards."""
import json
import sys
from pathlib import Path
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports


def write_mlx_checkpoint(folder, tensors, config):
    folder = Path(folder)
    folder.mkdir()
    mapping = {}
    for i, (key, value) in enumerate(sorted(tensors.items())):
        shard = f"part-{i}.safetensors"
        mx.save_safetensors(str(folder / shard), {key: value})
        mapping[key] = shard
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping, "metadata": {}}))
    return folder
