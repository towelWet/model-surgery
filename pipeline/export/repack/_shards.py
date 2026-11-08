# _shards.py — rewrite every shard: packed uint32 .weight 1-bit -> 2-bit, others copied as-is.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os

import numpy as np
from safetensors.numpy import load_file, save_file

from pipeline.export.repack._spread import repack_weight


def repack_shards(src, dst, weight_map):
    new_wm, done = {}, 0
    for sf in sorted(set(weight_map.values())):            # unique shard filenames
        tensors = load_file(os.path.join(src, sf))
        out = {}
        for name, arr in tensors.items():
            if name.endswith(".weight") and arr.dtype == np.uint32:
                out[name] = repack_weight(arr)              # only quantized weights are uint32
                done += 1
            else:
                out[name] = arr                             # scales/biases/float tensors unchanged
            new_wm[name] = sf
        save_file(out, os.path.join(dst, sf), metadata={"format": "mlx"})  # mlx loaders expect format tag
        print(f"  {sf}: {len(tensors)} tensors, {done} repacked total")
    return new_wm, done
