"""Nested or identical source/destination pairs rejected for every supplied source."""
import tempfile
from pathlib import Path
from pipeline.surgery.checkpoint.destination import validate_destination


def test_source_destination_overlap_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "outer-Tau" / "src-Tau"
        source.mkdir(parents=True)
        cases = [(source, FileExistsError),                       # identical: exists check first
                 (source / "Tau-nested", ValueError),             # dest nested in source
                 (source.parent, FileExistsError)]                # source nested in existing dest
        for dest, expected in cases:
            try:
                validate_destination(dest, sources=[source])
            except expected:
                pass
            else:
                raise AssertionError(f"accepted overlap {dest}")
        ok = Path(tmp) / "sibling-Tau"
        assert validate_destination(ok, sources=[source]) == ok.resolve()
