"""Checkpoint naming is general; safety checks must not impose a product brand."""
import tempfile
from pathlib import Path
from pipeline.surgery.checkpoint.destination import validate_destination


def test_general_checkpoint_names_are_resolved_without_creating_output():
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("child", "checkpoint-merged", "TAU-only"):
            dest = validate_destination(Path(tmp) / name)
            assert dest.name == name and dest.is_absolute() and not dest.exists()
        dest = validate_destination(Path(tmp) / "out-Tau-56L")
        assert dest.name == "out-Tau-56L" and dest.is_absolute()
        assert not dest.exists()
