"""NuSLERP slerps each row of the two task vectors separately, then adds them to base."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery.merge._ties.multi_np import merge_multi_np


def test_nuslerp_rowwise_on_deltas():
    base = np.ones((2, 2), dtype=np.float32)
    a = base + np.array([[1.0, 0.0], [2.0, 0.0]])            # row deltas along x
    b = base + np.array([[0.0, 1.0], [0.0, 2.0]])            # row deltas along y
    rng = np.random.default_rng(0)
    out = merge_multi_np(base, [a, b], "nuslerp", 1.0, [1.0, 1.0], 1.0, rng)
    r = np.sqrt(0.5)
    assert np.allclose(out - base, [[r, r], [2 * r, 2 * r]], atol=1e-5)  # each row on its own arc
    end = merge_multi_np(base, [a, b], "nuslerp", 1.0, [1.0, 0.0], 1.0, rng)
    assert np.allclose(end, a)                                # weight all on a -> a exactly
