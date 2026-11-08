# st/_header.py — parse one safetensors header without reading tensor data.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Format: 8-byte little-endian header length, JSON header, then flat data blob.
import json, struct


def header(path):
    """Return (tensor_table, data_start_offset) for one safetensors file."""
    with open(path, "rb") as f:
        size = f.seek(0, 2)
        f.seek(0)
        if size < 8:
            raise ValueError(f"{path}: truncated safetensors prefix")
        n = struct.unpack("<Q", f.read(8))[0]                # header byte length
        if n > size - 8 or n > 100_000_000:
            raise ValueError(f"{path}: invalid or oversized safetensors header")
        table = json.loads(f.read(n))
    table.pop("__metadata__", None)                          # not a tensor
    for name, entry in table.items():
        begin, end = entry["data_offsets"]
        if not 0 <= begin <= end <= size - 8 - n:
            raise ValueError(f"{path}: tensor {name} byte range outside shard")
    return table, 8 + n
