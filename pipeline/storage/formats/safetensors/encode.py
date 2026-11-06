"""Write one single-tensor safetensors file, streaming payload from a source region."""
import hashlib
import json
import os
import struct
from ...hashes.digest import CHUNK


def write_tensor_file(dest, name, dtype, shape, src_path, src_start, nbytes):
    """Build {name:{dtype,shape,offsets[0,nbytes]}} file; return payload sha256."""
    header = json.dumps({name: {"dtype": dtype, "shape": shape,
                                "data_offsets": [0, nbytes]}}).encode()
    digest = hashlib.sha256()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "xb") as out, open(src_path, "rb") as src:
        out.write(struct.pack("<Q", len(header)) + header)
        src.seek(src_start)
        left = nbytes
        while left:
            block = src.read(min(CHUNK, left))
            if not block:
                raise ValueError(f"{src_path}: short read for tensor {name}")
            out.write(block)
            digest.update(block)
            left -= len(block)
        out.flush()
        os.fsync(out.fileno())
    return digest.hexdigest()
