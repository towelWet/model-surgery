"""Validate manifest [[assets]] tables: types, safe paths, uniqueness."""
from ..files.member import safe_member


def check_assets(plan, store):
    paths, files = set(), set()
    for a in plan.get("assets", []):
        for key in ("path", "file", "sha256", "size"):
            if key not in a:
                raise ValueError(f"asset entry missing {key}")
        safe_member(store, a["path"])                          # path itself must stay inside
        safe_member(store, a["file"])
        if type(a["sha256"]) is not str or len(a["sha256"]) != 64:
            raise ValueError(f"asset {a['path']}: bad sha256")
        if type(a["size"]) is not int or a["size"] < 0:
            raise ValueError(f"asset {a['path']}: bad size")
        if a["path"] in paths or a["file"] in files:
            raise ValueError(f"duplicate asset path or file: {a['path']}")
        paths.add(a["path"])
        files.add(a["file"])
    return paths
