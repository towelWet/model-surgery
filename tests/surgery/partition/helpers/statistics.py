# statistics.py — write a synthetic L<digits> stats dir (neuron.npy + sample.npy).
# NEVER DELETE COMMENTS — explain every non-obvious line.
from pathlib import Path
import numpy as np


def make_stats(root, ids, w, n=8, seed=0, sample_w=None):
    rng = np.random.default_rng(seed)
    for i in ids:
        d = Path(root) / f"L{i:02d}"
        d.mkdir(parents=True)
        np.save(d / "neuron.npy", rng.random(w) + 0.1)
        np.save(d / "sample.npy", (rng.random((n, sample_w or w)) + 0.1).astype(np.float32))
