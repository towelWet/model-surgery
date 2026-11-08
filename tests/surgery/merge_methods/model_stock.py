"""Model Stock: parallel task vectors keep full average; orthogonal ones pull toward base."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery.merge._ties.multi_np import merge_multi_np


def test_model_stock_ratio_follows_angle():
    base = np.zeros(2, dtype=np.float32)
    rng = np.random.default_rng(0)
    same = merge_multi_np(base, [np.array([1.0, 0.0]), np.array([1.0, 0.0])], "model_stock", 1, [1, 1], 1, rng)
    assert np.allclose(same, [1.0, 0.0])                     # cos 1 -> t = 1: plain average
    ortho = merge_multi_np(base, [np.array([1.0, 0.0]), np.array([0.0, 1.0])], "model_stock", 1, [1, 1], 1, rng)
    assert np.allclose(ortho, [0.0, 0.0])                    # cos 0 -> t = 0: base
