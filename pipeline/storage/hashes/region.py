"""sha256 of a byte region inside a file, streamed in 1 MiB chunks."""
import hashlib
from .digest import CHUNK


def region_sha256(path, start, length):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        stream.seek(start)
        left = length
        while left:
            block = stream.read(min(CHUNK, left))
            if not block:
                raise ValueError(f"{path}: short read inside region")
            digest.update(block)
            left -= len(block)
    return digest.hexdigest()
