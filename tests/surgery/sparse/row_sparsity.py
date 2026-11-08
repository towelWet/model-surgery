"""Row mask zeroes exactly requested fraction in every row, lowest scores first."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np


def test_row_mask_exact_per_row():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.sparse._row_mask import row_mask
    score = np.array([[4.0, 1.0, 3.0, 2.0], [0.1, 0.4, 0.3, 0.2]])
    keep = row_mask(score, 0.5)
    assert keep.tolist() == [[True, False, True, False], [False, True, True, False]]
