# _deq_rows.py — dequantize ROWS sampled rows of one MLX-quantized tensor.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json, os
import numpy as np, mlx.core as mx

from pipeline.checkpoint.tensors import tensor_index
from _st_rows import rows

def deq(model, name, r0, count=256):
    idx = tensor_index(model)
    config = json.load(open(os.path.join(model, "config.json")))
    q = config.get("quantization", config.get("text_config", {}).get("quantization", {}))
    params = {**q, **q.get(name, {})}
    w, s, b = (rows(idx[f"{name}.{k}"], r0, count) for k in ("weight", "scales", "biases"))
    return np.array(mx.dequantize(mx.array(w), mx.array(s.astype(np.float32)), mx.array(b.astype(np.float32)),
                                  group_size=params["group_size"], bits=params["bits"]), dtype=np.float32)
