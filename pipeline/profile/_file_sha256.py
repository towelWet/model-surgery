# _file_sha256.py — streaming SHA-256 of a whole file (matches Hugging Face lfs_sha256).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import hashlib

CHUNK = 64 << 20                                             # 64 MB read window: flat RAM


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()
