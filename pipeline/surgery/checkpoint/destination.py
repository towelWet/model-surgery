"""Shared destination validation for checkpoint-writing pipeline entries."""
from pathlib import Path


def validate_destination(destination, sources=(), allow_existing=False):
    """Resolve any named checkpoint; reject existing paths and source overlap."""
    raw = Path(destination)
    if not str(destination).strip() or raw.name in ("", ".", ".."):
        raise ValueError("destination requires a nonempty directory name")
    if raw.is_symlink():                                     # link occupies the dir entry: same class as exists
        raise FileExistsError(f"destination is a symlink: {raw}")
    if raw.exists():
        if not allow_existing:
            raise FileExistsError(f"destination exists: {raw}")
        if not raw.is_dir():
            raise ValueError(f"resume destination is not a directory: {raw}")
    dest = raw.resolve()
    for source in sources:
        src = Path(source).resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"source missing: {src}")
        if src == dest or dest.is_relative_to(src) or src.is_relative_to(dest):
            raise ValueError(f"source/destination overlap: {src} vs {dest}")
    return dest
