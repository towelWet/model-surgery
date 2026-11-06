"""Validate manifest [[shards]] tables: types, safe paths, uniqueness."""
from ..files.member import safe_member


def check_shards(plan, store):
    seen = set()
    for s in plan.get("shards", []):
        for key in ("path", "header", "sha256", "size"):
            if key not in s:
                raise ValueError(f"shard entry missing {key}")
        safe_member(store, s["path"])
        safe_member(store, s["header"])
        if type(s["sha256"]) is not str or len(s["sha256"]) != 64:
            raise ValueError(f"shard {s['path']}: bad sha256")
        if type(s["size"]) is not int or s["size"] < 0:
            raise ValueError(f"shard {s['path']}: bad size")
        if s["path"] in seen:
            raise ValueError(f"duplicate shard path: {s['path']}")
        seen.add(s["path"])
    return seen
