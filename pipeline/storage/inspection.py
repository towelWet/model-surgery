"""inspect_store: manifest inventory; verify=True strictly checks every recorded file."""
import tomllib
from pathlib import Path


def inspect_store(store, verify=True):
    """Report store inventory; complete only with marker match AND zero defects."""
    store = Path(store).resolve()
    manifest = store / "manifest.toml"
    tx = store / "transaction.toml"
    if not manifest.is_file():
        return {"complete": False, "missing": ["manifest.toml"], "corrupt": [], "errors": [],
                "operation": tomllib.loads(tx.read_text()).get("operation")
                if tx.is_file() else None}
    plan = tomllib.loads(manifest.read_text())
    report = {"complete": False, "missing": [], "corrupt": [], "errors": [],
              "shards": len(plan.get("shards", [])), "tensors": len(plan.get("tensors", [])),
              "assets": len(plan.get("assets", [])),
              "payload_bytes": sum(t.get("nbytes", 0) for t in plan.get("tensors", [])),
              "verified": False, "source_fingerprint": plan.get("source_fingerprint"),
              "guarantees": plan.get("guarantees", "lossless-bytes")}
    from .manifest.validate import validate_manifest
    try:
        validate_manifest(plan, store)
    except ValueError as e:
        report["errors"].append(str(e))
    if verify and not report["errors"]:
        from .files.store_verify import verify_files
        report["missing"], report["corrupt"] = verify_files(store, plan)
        report["verified"] = True
    if (store / "complete.toml").is_file() and not report["errors"]:
        from .transaction.completion import check_complete
        try:
            check_complete(store)
            report["complete"] = report["verified"] and not report["missing"] and not report["corrupt"]
        except ValueError:
            report["corrupt"].append("complete.toml")
    return report
