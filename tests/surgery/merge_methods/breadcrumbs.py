"""Breadcrumbs drops biggest outlier and smallest deltas, keeps middle unscaled."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery.merge._ties.multi_np import merge_multi_np


def test_breadcrumbs_keeps_middle_band():
    base = np.zeros(10, dtype=np.float32)
    ft = np.arange(1, 11, dtype=np.float32)                  # deltas 1..10
    out = merge_multi_np(base, [ft], "breadcrumbs", 0.5, [1.0], 1.0, np.random.default_rng(0), gamma=0.1)
    # drop top 10% (10) and bottom 40% (1..4): keep 5..9 untouched
    assert out.tolist() == [0, 0, 0, 0, 5, 6, 7, 8, 9, 0]
