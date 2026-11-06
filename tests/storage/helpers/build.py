"""Write synthetic safetensors checkpoints as raw bytes; no model weights, no frameworks."""
import json
import struct
from pathlib import Path

DTYPE_BYTES = {"BF16": 2, "F32": 4, "U32": 4, "U8": 1}


def write_st(path, tensors, metadata=None):
    """tensors: [(name, dtype, shape, payload_bytes)]; payload sizes must match."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table, blobs, cursor = {}, [], 0
    for name, dtype, shape, payload in tensors:
        nbytes = DTYPE_BYTES[dtype]
        for dim in shape:
            nbytes *= dim
        assert nbytes == len(payload), f"{name}: payload {len(payload)} != {nbytes}"
        table[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + nbytes]}
        blobs.append(payload)
        cursor += nbytes
    if metadata:
        table["__metadata__"] = metadata
    header = json.dumps(table).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"".join(blobs))
    return path
