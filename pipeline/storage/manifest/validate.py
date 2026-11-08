"""Full semantic manifest validation: schema, tables, refs, collisions, fingerprint."""
from ..hashes.fingerprint import fingerprint
from .asset_entries import check_assets
from .shard_entries import check_shards
from .tensor_entries import check_tensors

RESERVED = {"transaction.toml", "manifest.toml", "complete.toml"}


def validate_manifest(plan, store):
    if type(plan.get("schema_version")) is not int or plan["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    if plan.get("kind") != "modular-safetensors":
        raise ValueError("kind must be modular-safetensors")
    if type(plan.get("source_fingerprint")) is not str:
        raise ValueError("missing source_fingerprint")
    for key in ("shards", "tensors", "assets"):
        if key not in plan:
            plan[key] = []                                   # omitted section == empty array of tables
        elif not isinstance(plan[key], list):
            raise ValueError(f"{key} must be an array of tables")
    shard_paths = check_shards(plan, store)
    check_tensors(plan, store, shard_paths)
    asset_paths = check_assets(plan, store)
    if shard_paths & asset_paths or (shard_paths | asset_paths) & RESERVED:
        raise ValueError("destination path collision with reserved or duplicate entries")
    declared = [(s["sha256"], s["path"]) for s in plan["shards"]]
    declared += [(a["sha256"], a["path"]) for a in plan["assets"]]
    if fingerprint(declared) != plan["source_fingerprint"]:
        raise ValueError("source_fingerprint does not match declared hashes")
    return plan
