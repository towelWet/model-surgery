"""TIES trims by magnitude, elects sign by summed mass, and averages only agreeing deltas."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery.merge._ties.multi_np import merge_multi_np


def test_ties_hand_example_matches_paper_rules():
    base = np.zeros(4, dtype=np.float32)
    a = np.array([3.0, -1.0, 0.1, 2.0], dtype=np.float32)    # fine-tune A task vector
    b = np.array([1.0, -4.0, -0.2, -1.0], dtype=np.float32)  # fine-tune B task vector
    merged = merge_multi_np(base, [a, b], "ties", 0.75, [1.0, 1.0], 1.0, np.random.default_rng(0))
    # trim 75%: A drops 0.1, B drops -0.2 -> A=[3,-1,0,2] B=[1,-4,0,-1]
    # sign mass: [4,-5,0,1] -> majority [+,-,+,+]; elem 2: no nonzero agrees -> 0
    # disjoint mean: [(3+1)/2, (-1-4)/2, 0, 2/1]
    assert np.allclose(merged, [2.0, -2.5, 0.0, 2.0])
    assert a[2] == np.float32(0.1) and b[3] == -1.0          # inputs untouched
