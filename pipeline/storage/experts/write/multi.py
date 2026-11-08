"""Write one multi-tensor safetensors file, streaming each entry's byte regions from sources.
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
"""
import hashlib
import json
import os
import struct
from pipeline.storage.hashes.digest import CHUNK


def write_multi(dest, entries):
    """entries: [{name, dtype, shape, regions:[(path,start,n)]}]. Returns {name: [region sha256]}."""
    table, cursor = {}, 0
    for e in entries:
        n = sum(r[2] for r in e["regions"])                     # tensor bytes = concat of its regions
        table[e["name"]] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [cursor, cursor + n]}
        cursor += n
    header = json.dumps(table, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)                         # 8-byte align payload (safetensors padding rule)
    shas = {}
    with open(dest, "xb") as out:                               # "x": never overwrite an existing file
        out.write(struct.pack("<Q", len(header)) + header)
        for e in entries:
            shas[e["name"]] = []
            for path, start, left in e["regions"]:
                digest = hashlib.sha256()
                with open(path, "rb") as src:
                    src.seek(start)
                    while left:
                        block = src.read(min(CHUNK, left))
                        if not block:
                            raise ValueError(f"{path}: short read for {e['name']}")
                        out.write(block)
                        digest.update(block)
                        left -= len(block)
                shas[e["name"]].append(digest.hexdigest())     # per-region hash: join checks each slice
        out.flush()
        os.fsync(out.fileno())                                  # durable before manifest claims it
    return shas
