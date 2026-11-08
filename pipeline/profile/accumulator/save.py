import os
import mlx.core as mx
import numpy as np


def save(self, out):
    for i, s in self.acc.items():
        d = os.path.join(out, f"L{i:02d}")
        os.makedirs(d, exist_ok=True)
        for key in ("xtx", "xtg", "neuron"):
            np.save(os.path.join(d, f"{key}.npy"), np.array(s[key]))
        np.save(os.path.join(d, "sample.npy"), np.array(mx.concatenate(s["sample"])))  # float16: ~3 GB total
