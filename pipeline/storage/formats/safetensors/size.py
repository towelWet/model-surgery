"""Expected byte size of one single-tensor safetensors file (prefix+header+payload)."""
import json


def tensor_file_size(t):
    header = json.dumps({t["name"]: {"dtype": t["dtype"], "shape": t["shape"],
                                     "data_offsets": [0, t["nbytes"]]}}).encode()
    return 8 + len(header) + t["nbytes"]
