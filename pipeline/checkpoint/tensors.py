"""Discover safetensors shards and validate an optional weight index, without MLX."""
import json
from pathlib import Path
from pipeline.profile.st._header import header


def tensor_index(folder):
    folder = Path(folder).resolve(strict=True)
    manifest = folder / "model.safetensors.index.json"
    mapping = json.loads(manifest.read_text())["weight_map"] if manifest.exists() else None
    shards = sorted(set(mapping.values())) if mapping else sorted(p.name for p in folder.glob("*.safetensors"))
    if not shards:
        raise ValueError("no safetensors shards found")
    result = {}
    for shard in shards:
        path = (folder / shard).resolve()
        if not path.is_relative_to(folder):
            raise ValueError(f"shard escapes checkpoint folder: {shard}")
        if not path.is_file():
            raise ValueError(f"missing shard: {shard}")
        entries, start = header(path)
        for name, entry in entries.items():
            if name in result:
                raise ValueError(f"duplicate tensor: {name}")
            result[name] = (path, entry, start)
    if mapping is not None:
        if set(mapping) != set(result):
            raise ValueError("weight index and shard tensor names disagree")
        for name, shard in mapping.items():
            if result[name][0] != (folder / shard).resolve():
                raise ValueError(f"weight index points at wrong shard for {name}")
    return result
