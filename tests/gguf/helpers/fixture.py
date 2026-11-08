"""Write tiny GGUF files from scratch (independent of read_gguf) so tests catch parser bugs."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import struct
import numpy as np
from pipeline.gguf.format.write.value import write_value


def write_fixture(path, arch, meta, tensors, align=32):
    """meta: {key: (vtype, value)}; tensors: {name: float32 ndarray in numpy (outer..inner) order}."""
    kv = [("general.architecture", 8, arch)] + [(f"{arch}.{k}", t, v) for k, (t, v) in meta.items()]
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kv)))
        for key, vtype, value in kv:
            write_value(f, 8, key)
            f.write(struct.pack("<I", vtype))
            write_value(f, vtype, value)
        offset = 0
        for name, arr in tensors.items():
            dims = list(reversed(arr.shape))                # ggml ne0 = numpy last axis
            write_value(f, 8, name)
            f.write(struct.pack(f"<I{len(dims)}Q", len(dims), *dims) + struct.pack("<IQ", 0, offset))
            offset += -(-arr.nbytes // align) * align
        f.write(b"\0" * (-f.tell() % align))
        for arr in tensors.values():
            f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
            f.write(b"\0" * (-f.tell() % align))
    return path
