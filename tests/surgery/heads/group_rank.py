"""Head-group scores follow real per-head output size; GQA groups sum their query heads."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np


def test_group_scores_rank_loud_group_first():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.heads._group_scores import group_scores
    rng = np.random.default_rng(5)
    o_in = rng.normal(size=(50, 4 * 2))                      # 4 query heads, head_dim 2
    o_in[:, 4:6] *= 10.0                                     # query head 2 is loud
    s = group_scores(o_in, np.eye(8), 4, 2, 2)               # identity o_proj, 2 KV groups (rep 2)
    assert s.shape == (2,) and s[1] > 3 * s[0]               # head 2 belongs to group 1
