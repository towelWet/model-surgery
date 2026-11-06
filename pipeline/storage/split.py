"""split_checkpoint: validated checkpoint -> modular-safetensors store (lossless byte contract)."""
import os
from pathlib import Path
from .plan import plan_split
from .partition.materialize import write_store
from .transaction.admission import admit_destination
from .transaction.begin import write_transaction
from .transaction.disk import disk_preflight
from .transaction.resume import resume_state


def split_checkpoint(source, destination, dry_run=False, resume=False):
    """Split source checkpoint into modular store; dry_run plans only (no writes)."""
    source = Path(source).resolve()
    dest = admit_destination(destination, sources=[source], resume=resume)  # before ANY I/O
    plan = plan_split(source)
    if dry_run:
        return {**plan, "dry_run": True, "hashes": "measured"}      # fingerprint needs hashes
    disk_preflight(dest, plan["required_bytes"])
    if dest.exists():
        resume_state(dest, "split", plan["source_fingerprint"])
        resuming = True
    else:
        os.mkdir(dest)
        write_transaction(dest, "split", plan["source_fingerprint"])
        resuming = False
    write_store(source, dest, plan, resuming, plan["_stamps"])
    return plan
