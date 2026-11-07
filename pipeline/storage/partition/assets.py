"""Preserve every checkpoint file not already represented as a tensor shard."""
from pathlib import Path
from .inventory import checkpoint_files
from ..files.member import safe_member
from ..hashes.digest import file_sha256


def plan_assets(source, shards=()):
    source = Path(source)
    names = set(checkpoint_files(source)) - set(shards)
    assets, hashes, stamps = [], [], {}
    for rel in sorted(set(names)):
        path = safe_member(source, rel)
        stat = path.stat()
        assets.append({"path": rel, "file": f"assets/{rel}",
                       "sha256": file_sha256(path), "size": stat.st_size})
        hashes.append((assets[-1]["sha256"], rel))
        stamps[rel] = (stat.st_size, stat.st_mtime_ns)
    return assets, hashes, stamps
