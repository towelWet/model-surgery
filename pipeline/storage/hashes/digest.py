"""Streaming sha256 of a file in 1 MiB chunks (never whole-file reads)."""
import hashlib

CHUNK = 1 << 20


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()
