"""Load manifest.toml and require schema_version 1 + modular-safetensors kind."""
import tomllib
from pathlib import Path
from ..transaction.completion import check_complete


def read_manifest(store):
    store = Path(store)
    check_complete(store)                                    # consumers refuse incomplete stores
    plan = tomllib.loads((store / "manifest.toml").read_text())
    if plan.get("schema_version") != 1 or plan.get("kind") != "modular-safetensors":
        raise ValueError("unsupported store schema or kind")
    return plan
