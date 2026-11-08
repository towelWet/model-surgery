"""Write one original shard: stored raw header bytes then tensor payloads in offset order."""
import hashlib
import os
from ..formats.safetensors.payload import payload_start
from ..files.member import safe_member
from ..hashes.digest import CHUNK


def join_shard(store, dest, shard, tensors):
    """tensors: manifest tensor entries for this shard; returns whole-file sha256."""
    target = safe_member(dest, shard["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with open(target, "xb") as out:
        blob = (safe_member(store, shard["header"])).read_bytes()
        out.write(blob)
        digest.update(blob)
        for t in sorted(tensors, key=lambda t: t["offsets"][0]):
            tfile = safe_member(store, t["file"])
            with open(tfile, "rb") as src:
                src.seek(payload_start(tfile))
                left = t["nbytes"]
                while left:
                    block = src.read(min(CHUNK, left))
                    if not block:
                        raise ValueError(f"{t['file']}: short payload for {t['name']}")
                    out.write(block)
                    digest.update(block)
                    left -= len(block)
        out.flush()
        os.fsync(out.fileno())
    return digest.hexdigest()
