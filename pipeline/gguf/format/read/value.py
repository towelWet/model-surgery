# read/value.py — decode one GGUF metadata value; arrays keep their element type for exact rewrite.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import struct

from ..tables import ARRAY, SCALARS, STRING


def read_value(f, vtype):
    if vtype == STRING:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "surrogateescape")   # surrogateescape: invalid bytes round-trip
    if vtype == ARRAY:
        etype, count = struct.unpack("<IQ", f.read(12))
        if etype in SCALARS:                                  # bulk-unpack numeric arrays (fast for 150k vocab)
            fmt = SCALARS[etype]
            size = struct.calcsize(fmt)
            items = list(struct.unpack("<" + fmt[1] * count, f.read(size * count)))
        else:
            items = [read_value(f, etype) for _ in range(count)]   # strings or nested arrays
        return {"type": etype, "items": items}
    if vtype not in SCALARS:
        raise ValueError(f"unknown GGUF value type {vtype}")
    fmt = SCALARS[vtype]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
