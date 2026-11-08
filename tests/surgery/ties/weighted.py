"""TIES applies per-model weights to sign mass and divisor, then lambda, then adds base."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery.merge._ties.multi_np import merge_multi_np


def test_ties_weights_lambda_and_base_offset():
    base = np.full(3, 10.0, dtype=np.float32)
    a, b = base + [2.0, 2.0, -2.0], base + [4.0, -1.0, -2.0]
    merged = merge_multi_np(base, [a, b], "ties", 1.0, [1.0, 3.0], 0.5, np.random.default_rng(0))
    # weighted: A=[2,2,-2] B=[12,-3,-6]; mass [14,-1,-8] -> [+,-,-]
    # elem0 (2+12)/4, elem1 -3/3, elem2 (-2-6)/4 -> [3.5,-1,-2] * 0.5 + 10
    assert np.allclose(merged, [11.75, 9.5, 9.0])
