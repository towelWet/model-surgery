"""Virtual shard reconstruction hash: header blob + ordered payloads, no writes."""
import hashlib
from ..files.member import safe_member
from ..hashes.digest import CHUNK
from ..formats.safetensors.payload import payload_start


def virtual_shard_sha256(store, shard, tensors):
    """Digest of what join would write for this shard; equals original file hash."""
    blob = safe_member(store, shard["header"]).read_bytes()
    digest = hashlib.sha256(blob)
    for t in sorted(tensors, key=lambda t: t["offsets"][0]):
        path = safe_member(store, t["file"])
        with open(path, "rb") as src:
            src.seek(payload_start(path))
            left = t["nbytes"]
            while left:
                block = src.read(min(CHUNK, left))
                if not block:
                    raise ValueError(f"{t['file']}: short payload for {t['name']}")
                digest.update(block)
                left -= len(block)
    return digest.hexdigest()
