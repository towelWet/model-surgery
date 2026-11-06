"""Every supplied source must be an existing directory; nothing is created."""
import tempfile
from pathlib import Path
from pipeline.surgery.checkpoint.destination import validate_destination


def test_missing_source_rejected_without_writes():
    with tempfile.TemporaryDirectory() as tmp:
        source, dest = Path(tmp) / "ghost", Path(tmp) / "out-Tau"
        try:
            validate_destination(dest, sources=[source])
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("accepted missing source")
        assert not dest.exists()                          # no writes on failure
