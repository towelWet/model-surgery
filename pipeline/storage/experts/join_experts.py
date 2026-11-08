"""join_experts: expert-split store -> one model.safetensors in original (stacked) layout, sha-checked.
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
"""
import tomllib
from pathlib import Path
from pipeline.storage.experts.copy_assets import copy_assets
from pipeline.storage.experts.read_tensors import read_tensors
from pipeline.storage.experts.write.multi import write_multi


def join_experts(store, dest):
    """Restack .eNNN rows by source name in original order; raise if any piece hash differs."""
    store, dest = Path(store).resolve(), Path(dest).resolve()
    man = store / "moe_split.toml"
    if not man.exists():
        raise ValueError(f"{store}: no moe_split.toml (incomplete or not an expert-split store)")
    pieces = tomllib.loads(man.read_text())["pieces"]
    where = {(t["path"], t["name"]): t for t in read_tensors(store)}   # live byte regions in store files
    merged = {}
    for p in sorted(pieces, key=lambda p: (p["order"], p["row"])):   # rows 0..E-1 in expert order
        t = where[(str(store / p["file"]), p["name"])]
        e = merged.setdefault(p["order"], {"name": p["source"], "dtype": p["dtype"],
                                           "shape": p["source_shape"], "regions": [], "expect": []})
        e["regions"].append((t["path"], t["start"], t["nbytes"]))
        e["expect"].append(p["sha256"])
    entries = [merged[k] for k in sorted(merged)]
    dest.mkdir()                                                     # refuses existing dest: no overwrite
    shas = write_multi(dest / "model.safetensors", entries)
    bad = [e["name"] for e in entries if shas[e["name"]] != e["expect"]]
    if bad:
        raise ValueError(f"piece hash mismatch for {len(bad)} tensors, first {bad[0]}; output untrusted")
    copy_assets(store, dest)
    return {"tensors": len(entries), "dest": str(dest)}
