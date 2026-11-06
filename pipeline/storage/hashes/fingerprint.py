"""source_fingerprint: sha256 over sorted 'sha256  path' lines for shards + assets."""
import hashlib


def fingerprint(entries):
    digest = hashlib.sha256()
    for line in sorted(f"{sha}  {rel}" for sha, rel in entries):
        digest.update(line.encode() + b"\n")
    return digest.hexdigest()
