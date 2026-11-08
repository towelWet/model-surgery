"""Read one F32 tensor back from a GGUF as numpy (outer..inner order) for value checks."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.gguf.format.read.gguf import read_gguf


def read_tensor(path, name):
    t = next(x for x in read_gguf(path)["tensors"] if x["name"] == name)
    (pos, n), = t["chunks"]
    with open(path, "rb") as f:
        f.seek(pos)
        return np.frombuffer(f.read(n), dtype=np.float32).reshape(list(reversed(t["dims"])))
