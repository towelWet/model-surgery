# _safetensors_header.py — read safetensors JSON header without mapping weights.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import struct


def safetensors_header(path):
    """Return the header dict (tensor names → dtype/shape/offsets). No tensor bytes."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]  # little-endian u64 header length
        raw = f.read(n)
    return json.loads(raw)
