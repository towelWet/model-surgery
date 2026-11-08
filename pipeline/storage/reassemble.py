"""reassemble_checkpoint: verified modular store -> original checkpoint bytes (lossless)."""
import os
import tomllib
from pathlib import Path
from .inspection import inspect_store
from .manifest.validate import validate_manifest
from .reconstruction.join_all import join_all
from .transaction.admission import admit_destination
from .transaction.begin import write_transaction
from .transaction.disk import disk_preflight
from .transaction.resume import resume_state


def reassemble_checkpoint(store, destination, resume=False):
    """Join a complete verified store back into the original checkpoint layout."""
    store = Path(store).resolve()
    dest = admit_destination(destination, sources=[store], resume=resume)   # before ANY I/O
    report = inspect_store(store, verify=True)
    if not report["complete"] or report["missing"] or report["corrupt"]:
        raise ValueError(f"store not join-ready: {report}")
    plan = validate_manifest(tomllib.loads((store / "manifest.toml").read_text()), store)
    required = sum(s["size"] for s in plan["shards"]) + sum(a["size"] for a in plan["assets"])
    disk_preflight(dest, required + 8192)
    if dest.exists():
        resume_state(dest, "join", plan["source_fingerprint"])
        resuming = True
    else:
        os.mkdir(dest)
        write_transaction(dest, "join", plan["source_fingerprint"])
        resuming = False
    join_all(store, dest, plan, resuming)
    return {"shards": len(plan["shards"]), "tensors": len(plan["tensors"]),
            "guarantees": "lossless-bytes", "source_fingerprint": plan["source_fingerprint"]}
