"""Plan a modular-safetensors split of one checkpoint directory (lossless byte contract only)."""
from pathlib import Path
from .formats.safetensors.size import tensor_file_size
from .formats.safetensors.widths import dtype_bytes
from .hashes.fingerprint import fingerprint
from .manifest.emit import manifest_text
from .partition.assets import plan_assets
from .partition.rels import shard_rels
from .partition.shards import plan_shards


def plan_split(source):
    """Validate source and return the full split plan dict (no destination writes)."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source checkpoint missing: {source}")
    rels, mapping = shard_rels(source)
    if mapping is not None:                                  # index agreement check reuses tensor_index
        from pipeline.checkpoint.tensors import tensor_index
        tensor_index(source)
    dtype_map = dtype_bytes()
    shards, tensors, shard_hashes, shard_stamps = plan_shards(source, rels, dtype_map)
    assets, asset_hashes, asset_stamps = plan_assets(source, rels)
    plan = {"schema_version": 1, "kind": "modular-safetensors", "guarantees": "lossless-bytes",
            "source_fingerprint": fingerprint(shard_hashes + asset_hashes),
            "shards": shards, "tensors": tensors, "assets": assets,
            "_stamps": {**shard_stamps, **asset_stamps}}
    from .manifest.validate import validate_manifest
    validate_manifest(plan, source)                          # reject collisions before destination writes
    manifest_len = len(manifest_text(plan).encode())
    plan["required_bytes"] = (sum(tensor_file_size(t) for t in tensors)
                              + sum(s["_header_bytes"] for s in shards)
                              + sum(a["size"] for a in assets)
                              + manifest_len + 4096)         # transaction + complete headroom
    return plan
