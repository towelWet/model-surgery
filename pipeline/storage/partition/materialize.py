"""Write all store files for a planned split, then manifest, then completion marker LAST."""
from ..files.member import safe_member
from ..files.exclusive import write_xb
from ..files.partial import verify_partial
from ..files.transfer import copy_file
from ..manifest.emit import manifest_text
from ..transaction.commit import write_complete
from .hbytes import header_bytes
from .stability import check_source_stable
from .tensors_out import write_tensors


def write_store(source, dest, plan, resuming, stamps):
    for shard in plan["shards"]:
        target = safe_member(dest, shard["header"])
        blob = header_bytes(source, shard["path"])
        if target.exists():
            if not resuming or target.read_bytes() != blob:    # exact bytes, not just length
                raise ValueError(f"damaged partial file (no overwrite allowed): {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            write_xb(target, blob)
    write_tensors(source, dest, plan, resuming)
    for a in plan["assets"]:
        target = safe_member(dest, a["file"])
        if not (resuming and verify_partial(target, a["size"], a["sha256"])):
            if copy_file(safe_member(source, a["path"]), target) != a["sha256"]:
                raise ValueError(f"asset hash mismatch: {a['path']}")
    check_source_stable(source, plan, stamps)
    blob = manifest_text(plan).encode()
    manifest = dest / "manifest.toml"
    if manifest.exists():
        if manifest.read_bytes() != blob:
            raise ValueError("existing manifest.toml disagrees with recomputed plan")
    else:
        write_xb(manifest, blob)
    write_complete(dest)
