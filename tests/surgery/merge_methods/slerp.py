"""SLERP hits endpoints at t=0/1 and keeps unit norm between orthogonal unit tensors."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery.merge._ties.multi_np import merge_multi_np


def test_slerp_endpoints_and_arc():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    rng = np.random.default_rng(0)
    ends = [merge_multi_np(a, [b], "slerp", 1.0, [1.0], t, rng) for t in (0.0, 1.0)]
    assert np.allclose(ends[0], a) and np.allclose(ends[1], b)
    mid = merge_multi_np(a, [b], "slerp", 1.0, [1.0], 0.5, rng)
    assert np.allclose(mid, [np.sqrt(0.5), np.sqrt(0.5)])    # on arc, unit norm (lerp gives 0.707 norm)
    assert np.allclose(merge_multi_np(a, [a * 2], "slerp", 1.0, [1.0], 0.5, rng), a * 1.5)  # parallel -> lerp
