"""Validate an interrupted store for --resume: matching op+fingerprint, never completed."""
import tomllib
from pathlib import Path


def resume_state(store, operation, fingerprint):
    store = Path(store)
    tx = store / "transaction.toml"
    if not tx.is_file():
        raise ValueError("no interrupted transaction to resume")
    data = tomllib.loads(tx.read_text())
    if data.get("operation") != operation:
        raise ValueError(f"transaction is {data.get('operation')!r}, not {operation!r}")
    if data.get("source_fingerprint") != fingerprint:
        raise ValueError("source changed since interrupted run — refusing resume")
    if (store / "complete.toml").exists():
        raise ValueError("store already complete; resume refused")
    return data
