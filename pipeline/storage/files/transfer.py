"""Exclusive streamed file copy ('xb', 1 MiB chunks, fsync); returns destination sha256."""
import hashlib
import os
from ..hashes.digest import CHUNK


def copy_file(src, dest):
    digest = hashlib.sha256()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, open(dest, "xb") as out:
        for block in iter(lambda: fin.read(CHUNK), b""):
            out.write(block)
            digest.update(block)
        out.flush()
        os.fsync(out.fileno())
    return digest.hexdigest()
