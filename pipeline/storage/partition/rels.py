"""List shard relative paths: strict index weight_map when present, else recursive scan."""
import json
from pathlib import Path
from ..files.member import safe_member
from ..formats.safetensors.duplicates import strict_object


def shard_rels(source):
    source = Path(source)
    index = source / "model.safetensors.index.json"
    if index.exists():
        if index.is_symlink() and not index.resolve().is_relative_to(source.resolve()):
            raise ValueError("index symlink escapes checkpoint")
        table = json.loads(index.read_text(), object_pairs_hook=strict_object)
        mapping = table.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("weight index missing or empty weight_map")
        if any(type(k) is not str or type(v) is not str for k, v in mapping.items()):
            raise ValueError("weight_map must map string names to string shard paths")
        for rel in set(mapping.values()):
            safe_member(source, rel)
        return sorted(set(mapping.values())), mapping
    shards = sorted(str(p.relative_to(source)) for p in source.rglob("*.safetensors"))
    if not shards:
        raise ValueError("no safetensors shards found")
    return shards, None
