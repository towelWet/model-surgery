"""2:4 mask keeps two highest of every four inputs and refuses widths not divisible by M."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np


def test_nm_mask_two_of_four():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.sparse._nm_mask import nm_mask
    keep = nm_mask(np.array([[1.0, 5.0, 3.0, 2.0, 9.0, 8.0, 0.0, 7.0]]), 2, 4)
    assert keep.tolist() == [[False, True, True, False, True, True, False, False]]
    try:
        nm_mask(np.ones((1, 6)), 2, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted width 6 for 2:4")
