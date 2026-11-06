"""Fingerprint eval source so changed suites cannot silently reuse old scores."""
import hashlib
from pathlib import Path


def fingerprint(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    sources = set(root.rglob("*.py"))
    profile = root.parent / "profile"
    sources.update(profile.rglob("*.py"))
    sources.update((profile / "schema").glob("*.json"))
    for path in sorted(sources):
        if "__pycache__" in path.parts:
            continue
        name = path.relative_to(root.parent).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(name).to_bytes(8, "little") + name)
        digest.update(len(payload).to_bytes(8, "little") + payload)
    return digest.hexdigest()
