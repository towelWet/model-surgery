"""Ancillary asset plan: config.json, weight index, discovered tokenizer files + drift stamps."""
from pathlib import Path
from pipeline.surgery.shard.tokenizer.discover import discover_assets
from ..files.member import safe_member
from ..hashes.digest import file_sha256


def plan_assets(source):
    source = Path(source)
    names = ["config.json", "model.safetensors.index.json"]
    names += [str(p) for p in discover_assets(source)]
    assets, hashes, stamps = [], [], {}
    for rel in sorted(set(names)):
        path = safe_member(source, rel)
        if not path.is_file():
            continue                                        # optional assets may be absent
        stat = path.stat()
        assets.append({"path": rel, "file": f"assets/{rel}",
                       "sha256": file_sha256(path), "size": stat.st_size})
        hashes.append((assets[-1]["sha256"], rel))
        stamps[rel] = (stat.st_size, stat.st_mtime_ns)
    return assets, hashes, stamps
