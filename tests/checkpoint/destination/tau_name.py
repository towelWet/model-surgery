"""Destination basename must contain case-sensitive 'Tau' and resolve to a Path."""
import tempfile
from pathlib import Path
from pipeline.surgery.checkpoint.destination import validate_destination


def test_tau_basename_required_and_resolved():
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("child", "lowertau", "TAU-only"):
            try:
                validate_destination(Path(tmp) / name)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted non-Tau name {name}")
        dest = validate_destination(Path(tmp) / "out-Tau-56L")
        assert dest.name == "out-Tau-56L" and dest.is_absolute()
        assert not dest.exists()
