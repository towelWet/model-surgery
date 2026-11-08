"""SparseGPT port: hits sparsity / 2:4 pattern and beats magnitude on output error with correlated inputs."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np


def test_sparsegpt_sparsity_pattern_and_error():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.sparse._sparsegpt_matrix import sparsegpt_matrix
    rng = np.random.default_rng(0)
    mix = rng.normal(size=(64, 64))
    x = rng.normal(size=(2000, 64)) @ mix                     # correlated inputs: where OBS update helps
    w = rng.normal(size=(32, 64))
    h = 2.0 * x.T @ x / len(x)
    p = sparsegpt_matrix(w, h, 0.5, block=16)
    assert abs((p == 0).mean() - 0.5) < 0.02                  # threshold per block ~ target fraction
    mag = np.where(np.abs(w) >= np.median(np.abs(w), axis=1, keepdims=True), w, 0.0)
    err = lambda m: np.linalg.norm(x @ (w - m).T)
    assert err(p) < 0.8 * err(mag)                            # error compensation clearly wins
    q = sparsegpt_matrix(w, h, 0.5, nm=(2, 4), block=16)
    assert ((q.reshape(32, 16, 4) != 0).sum(-1) <= 2).all()   # at most 2 of every 4 kept
