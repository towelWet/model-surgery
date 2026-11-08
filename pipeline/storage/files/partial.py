"""Resume check: an existing partial file must match expected size(+sha256) or fail clearly."""
from pathlib import Path
from ..hashes.digest import file_sha256


def verify_partial(path, size, sha256=None):
    """Return True if file exists and matches; False if missing; raise on damage."""
    path = Path(path)
    if not path.exists():
        return False
    if path.stat().st_size != size:
        raise ValueError(f"damaged partial file (no overwrite allowed): {path}")
    if sha256 is not None and file_sha256(path) != sha256:
        raise ValueError(f"damaged partial file (no overwrite allowed): {path}")
    return True
