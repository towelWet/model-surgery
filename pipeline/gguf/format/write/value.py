# write/value.py — encode one GGUF metadata value; inverse of read_value (byte-exact).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import struct

from ..tables import ARRAY, SCALARS, STRING


def write_value(f, vtype, value):
    if vtype == STRING:
        raw = value.encode("utf-8", "surrogateescape")
        f.write(struct.pack("<Q", len(raw)) + raw)
    elif vtype == ARRAY:
        etype, items = value["type"], value["items"]
        f.write(struct.pack("<IQ", etype, len(items)))
        if etype in SCALARS:
            f.write(struct.pack("<" + SCALARS[etype][1] * len(items), *items))
        else:
            for item in items:
                write_value(f, etype, item)
    else:
        f.write(struct.pack(SCALARS[vtype], value))
