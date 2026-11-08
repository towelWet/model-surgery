"""List every tensor of a checkpoint dir as byte regions (no weight decoding, no frameworks).
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
"""
from pathlib import Path
from pipeline.storage.formats.safetensors.header import raw_header


def read_tensors(src):
    """Return [{name, dtype, shape, path, start, nbytes}] in shard-then-offset order."""
    out, seen = [], set()
    for path in sorted(Path(src).glob("*.safetensors")):     # top-level shards only; index.json not trusted
        if path.name.startswith("._"):
            continue                                          # macOS AppleDouble metadata on exFAT, not a shard
        raw, table, _ = raw_header(path)
        base = 8 + len(raw)                                   # payload starts after u64 length + header
        rows = [(k, v) for k, v in table.items() if k != "__metadata__"]
        for name, info in sorted(rows, key=lambda kv: kv[1]["data_offsets"][0]):
            if name in seen:
                raise ValueError(f"{path}: tensor {name} appears in two shards")
            seen.add(name)
            lo, hi = info["data_offsets"]
            out.append({"name": name, "dtype": info["dtype"], "shape": list(info["shape"]),
                        "path": str(path), "start": base + lo, "nbytes": hi - lo})
    if not out:
        raise ValueError(f"{src}: no *.safetensors tensors found")
    return out
