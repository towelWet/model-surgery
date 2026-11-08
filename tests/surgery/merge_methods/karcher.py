"""Karcher mean of unit tensors lies on sphere between them; equals slerp midpoint for two."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery.merge._ties.multi_np import merge_multi_np


def test_karcher_midpoint_and_three_axes():
    base = np.zeros(3, dtype=np.float32)
    rng = np.random.default_rng(0)
    x, y, z = np.eye(3, dtype=np.float32)
    two = merge_multi_np(base, [x, y], "karcher", 1.0, [1.0, 1.0], 1.0, rng)
    assert np.allclose(two, [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-5)  # geodesic midpoint
    three = merge_multi_np(base, [x, y, z], "karcher", 1.0, [1.0, 1.0, 1.0], 1.0, rng)
    assert np.allclose(three, np.full(3, 1 / np.sqrt(3)), atol=1e-5)       # symmetric centre, unit norm
    scaled = merge_multi_np(base, [x * 2, y * 4], "karcher", 1.0, [1.0, 1.0], 1.0, rng)
    assert np.isclose(np.linalg.norm(scaled), 3.0)                           # norm = mean of input norms
