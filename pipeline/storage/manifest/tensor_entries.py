"""Validate manifest [[tensors]] tables: fields, shard refs, offsets, unique names/files."""
from ..files.member import safe_member
from ..formats.safetensors.widths import dtype_bytes


def check_tensors(plan, store, shard_paths):
    names, files = set(), set()
    widths = dtype_bytes()
    for t in plan.get("tensors", []):
        for key in ("name", "dtype", "shape", "shard", "offsets", "file", "sha256", "nbytes"):
            if key not in t:
                raise ValueError(f"tensor entry missing {key}")
        if t["shard"] not in shard_paths:
            raise ValueError(f"tensor {t['name']}: unknown shard {t['shard']}")
        safe_member(store, t["file"])
        shape = t["shape"]
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f"tensor {t['name']}: invalid shape")
        begin, end = t["offsets"]
        if not (type(begin) is int and type(end) is int and 0 <= begin <= end):
            raise ValueError(f"tensor {t['name']}: invalid offsets")
        if t["nbytes"] != end - begin or type(t["nbytes"]) is not int:
            raise ValueError(f"tensor {t['name']}: nbytes disagrees with offsets")
        if t["dtype"] not in widths:
            raise ValueError(f"tensor {t['name']}: unknown dtype {t['dtype']}")
        count = 1
        for d in shape:
            count *= d
        if t["nbytes"] != count * widths[t["dtype"]]:
            raise ValueError(f"tensor {t['name']}: nbytes disagrees with dtype+shape")
        if t["name"] in names or t["file"] in files:
            raise ValueError(f"duplicate tensor name or file: {t['name']}")
        names.add(t["name"])
        files.add(t["file"])
    return names, files
