# read/gguf.py — parse GGUF header: metadata, tensor table, absolute payload offsets. Reads no weights.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import struct

from .value import read_value
from ..tables import DEFAULT_ALIGNMENT, MAGIC
from ..tensor_nbytes import tensor_nbytes


def read_gguf(path):
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != MAGIC or version not in (2, 3):
            raise ValueError(f"{path}: not a GGUF v2/v3 file")
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        kv = []                                               # ordered: rewrite keeps original order
        for _ in range(n_kv):
            key = read_value(f, 8)
            vtype = struct.unpack("<I", f.read(4))[0]
            kv.append({"key": key, "type": vtype, "value": read_value(f, vtype)})
        tensors = []
        for _ in range(n_tensors):
            name = read_value(f, 8)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))  # ne0 (fastest) first
            ggml_type, offset = struct.unpack("<IQ", f.read(12))
            tensors.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset,
                            "nbytes": tensor_nbytes(dims, ggml_type)})
        align = next((e["value"] for e in kv if e["key"] == "general.alignment"), DEFAULT_ALIGNMENT)
        start = -(-f.tell() // align) * align               # payload section begins at next aligned byte
    for t in tensors:
        t["chunks"] = [(start + t["offset"], t["nbytes"])]  # absolute byte ranges to copy on rewrite
    return {"version": version, "kv": kv, "tensors": tensors, "alignment": align, "data_start": start}
