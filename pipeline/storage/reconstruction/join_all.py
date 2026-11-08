"""Write every shard+asset for a join, then join manifest, then completion LAST."""
from ..files.exclusive import write_xb
from ..files.member import safe_member
from ..files.partial import verify_partial
from ..files.transfer import copy_file
from ..manifest.join_emit import join_manifest_text
from ..transaction.commit import write_complete
from .shard import join_shard


def join_all(store, dest, plan, resuming):
    for shard in plan["shards"]:
        target = safe_member(dest, shard["path"])
        tensors = [t for t in plan["tensors"] if t["shard"] == shard["path"]]
        if resuming and verify_partial(target, shard["size"], shard["sha256"]):
            continue
        if join_shard(store, dest, shard, tensors) != shard["sha256"]:
            raise ValueError(f"reassembled shard hash mismatch: {shard['path']}")
    for a in plan["assets"]:
        target = safe_member(dest, a["path"])
        if not (resuming and verify_partial(target, a["size"], a["sha256"])):
            if copy_file(safe_member(store, a["file"]), target) != a["sha256"]:
                raise ValueError(f"asset hash mismatch: {a['path']}")
    blob = join_manifest_text(plan).encode()
    manifest = dest / "manifest.toml"
    if manifest.exists():
        if manifest.read_bytes() != blob:
            raise ValueError("existing manifest.toml disagrees with expected join manifest")
    else:
        write_xb(manifest, blob)
    write_complete(dest)
