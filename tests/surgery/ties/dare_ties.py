"""DARE-TIES rescales random survivors before sign election; bad methods/densities/weights refused."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery.merge._ties.multi_np import merge_multi_np


def test_dare_ties_and_bad_inputs():
    base = np.zeros(10_000, dtype=np.float32)
    ones = np.ones_like(base)
    merged = merge_multi_np(base, [ones, ones], "dare_ties", 0.5, [1.0, 1.0], 1.0, np.random.default_rng(3))
    assert set(np.unique(merged)) <= {0.0, 2.0}               # survivors rescaled 1/0.5, agree, mean
    assert abs(float(merged.mean()) - 1.5) < 0.05            # P(any kept)=0.75 -> E=0.75*2
    for bad in (("mean", 0.5, [1.0, 1.0]), ("ties", 0.0, [1.0, 1.0]), ("ties", 0.5, [1.0])):
        try:
            merge_multi_np(base, [ones, ones], bad[0], bad[1], bad[2], 1.0, np.random.default_rng(0))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
