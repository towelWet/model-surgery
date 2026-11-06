"""Resolve a store/source-relative path, rejecting traversal and escaping symlinks."""
from pathlib import Path


def safe_member(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe relative path: {relative}")
    resolved = (Path(root) / relative).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f"path escapes root: {relative}")
    return resolved
