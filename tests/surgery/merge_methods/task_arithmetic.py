"""Task arithmetic adds weighted task vectors to base with no drop (Ilharco et al. 2023)."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery.merge._ties.multi_np import merge_multi_np


def test_task_arithmetic_hand_example():
    base = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    a = np.array([2.0, 1.0, 0.0], dtype=np.float32)          # task vector [1, 0, -1]
    b = np.array([1.0, 3.0, 1.0], dtype=np.float32)          # task vector [0, 2, 0]
    out = merge_multi_np(base, [a, b], "task_arithmetic", 0.1, [1.0, 0.5], 2.0, np.random.default_rng(0))
    # lambda 2 * (1*[1,0,-1] + 0.5*[0,2,0]) + base = [3, 3, -1]; density ignored (no drop)
    assert np.allclose(out, [3.0, 3.0, -1.0])
