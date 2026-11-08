"""Slice spec keeps order, expands inclusive ranges, and refuses out-of-range layers."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path


def test_parse_slices_order_and_bounds():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.passthrough._parse_slices import parse_slices
    assert parse_slices("1:2-3,0:0", [4, 4]) == [(1, 2), (1, 3), (0, 0)]
    try:
        parse_slices("0:0-4", [4])
    except ValueError:
        pass
    else:
        raise AssertionError("accepted layer 4 of 4-layer model")
