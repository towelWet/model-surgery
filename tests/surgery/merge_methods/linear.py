"""Linear (Model Soup) averages whole fine-tuned weights by normalised weights."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery.merge._ties.multi_np import merge_multi_np


def test_linear_soup_weighted_average():
    base = np.zeros(2, dtype=np.float32)                     # layout only; not averaged in
    a = np.array([4.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 8.0], dtype=np.float32)
    out = merge_multi_np(base, [a, b], "linear", 1.0, [3.0, 1.0], 1.0, np.random.default_rng(0))
    assert np.allclose(out, [3.0, 2.0])                      # (3*a + 1*b) / 4
