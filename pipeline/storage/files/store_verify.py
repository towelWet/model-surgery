"""Strict whole-store file verification: headers, tensor headers+payloads, assets, virtual join."""
from .member import safe_member
from .tensor_verify import verify_tensor_file
from ..hashes.digest import file_sha256
from ..reconstruction.virtual import virtual_shard_sha256


def verify_files(store, plan):
    """Return (missing, corrupt) relative paths; every recorded file fully checked."""
    missing, corrupt = [], []
    shards = {s["path"]: s for s in plan["shards"]}
    for s in plan["shards"]:
        hpath = safe_member(store, s["header"])
        if not hpath.is_file():
            missing.append(s["header"])
            continue
        tensors = [t for t in plan["tensors"] if t["shard"] == s["path"]]
        if any(not safe_member(store, t["file"]).is_file() for t in tensors):
            missing += [t["file"] for t in tensors
                        if not safe_member(store, t["file"]).is_file()]
            continue
        if virtual_shard_sha256(store, s, tensors) != s["sha256"]:
            corrupt.append(s["header"])                      # header or payload chain mismatch
    for t in plan["tensors"]:
        path = safe_member(store, t["file"])
        if not path.is_file():
            if t["file"] not in missing:
                missing.append(t["file"])
            continue
        try:
            verify_tensor_file(path, t)
        except ValueError:
            corrupt.append(t["file"])
    for a in plan["assets"]:
        path = safe_member(store, a["file"])
        if not path.is_file():
            missing.append(a["file"])
        elif path.stat().st_size != a["size"] or file_sha256(path) != a["sha256"]:
            corrupt.append(a["file"])
    return missing, corrupt
