"""DELLA keeps large deltas more often than small ones and stays unbiased on average."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery.merge._ties.multi_np import merge_multi_np


def test_della_magnitude_bias_and_unbiased_mean():
    base = np.zeros(20000, dtype=np.float32)
    ft = np.where(np.arange(20000) < 10000, 0.1, 1.0).astype(np.float32)  # half small, half large
    out = merge_multi_np(base, [ft], "della_linear", 0.5, [1.0], 1.0, np.random.default_rng(1), epsilon=0.4)
    small_kept, big_kept = (out[:10000] != 0).mean(), (out[10000:] != 0).mean()
    assert big_kept > small_kept + 0.3                       # keep prob ~0.3 vs ~0.7
    assert abs(out[10000:].mean() - 1.0) < 0.05              # rescale by 1/p: expectation preserved
