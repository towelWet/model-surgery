"""SVD factors: exact at full rank; ASVD trades error onto channels inputs barely use."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np


def test_svd_and_asvd():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.lowrank._svd_factor import svd_factor
    from pipeline.surgery.lowrank._keep_rank import keep_rank
    rng = np.random.default_rng(0)
    w = rng.normal(size=(8, 6)).astype(np.float32)
    a, b = svd_factor(w, 6)
    assert np.allclose(a @ b, w, atol=1e-5)                   # full rank reproduces W
    s = np.array([10, 10, 10, 0.1, 0.1, 0.1], dtype=np.float32)  # first 3 inputs carry real signal
    x = rng.normal(size=(64, 6)).astype(np.float32) * s      # activations shaped like scale
    err = lambda a_, b_: np.linalg.norm(x @ (w - a_ @ b_).T)  # error on real outputs, not on W
    assert err(*svd_factor(w, 3, s)) < err(*svd_factor(w, 3))  # ASVD wins on activation-weighted error
    assert keep_rank(1024, 3072, 0.5) == 384 and keep_rank(64, 64, 0.1) == 64  # 64-multiple, floor one group
