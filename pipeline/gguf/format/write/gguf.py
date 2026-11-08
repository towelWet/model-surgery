# write/gguf.py — write GGUF v3 from parsed model; tensor bytes stream from source chunks.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import struct

from ..tables import MAGIC, STRING
from .value import write_value

COPY = 64 << 20                                           # 64 MiB copy window: bounded RAM for 20 GB files


def write_gguf(source, dest, model):
    if os.path.exists(dest):
        raise FileExistsError(f"{dest} exists; refusing to overwrite")
    align, tensors = model["alignment"], model["tensors"]
    tmp = dest + ".partial"                                 # rename at end: no half-written dest
    with open(source, "rb") as src, open(tmp, "xb") as out:
        out.write(struct.pack("<IIQQ", MAGIC, 3, len(tensors), len(model["kv"])))
        for e in model["kv"]:
            write_value(out, STRING, e["key"])
            out.write(struct.pack("<I", e["type"]))
            write_value(out, e["type"], e["value"])
        offset = 0
        for t in tensors:                                   # relative offsets, each aligned
            write_value(out, STRING, t["name"])
            out.write(struct.pack(f"<I{len(t['dims'])}Q", len(t["dims"]), *t["dims"]))
            out.write(struct.pack("<IQ", t["type"], offset))
            offset += -(-sum(n for _, n in t["chunks"]) // align) * align
        out.write(b"\0" * (-out.tell() % align))            # pad header to payload start
        for t in tensors:
            for pos, n in t["chunks"]:
                src.seek(pos)
                while n:
                    block = src.read(min(n, COPY))
                    if not block:
                        raise ValueError(f"source truncated inside {t['name']}")
                    out.write(block)
                    n -= len(block)
            out.write(b"\0" * (-out.tell() % align))        # pad tensor to next aligned slot
    os.replace(tmp, dest)
