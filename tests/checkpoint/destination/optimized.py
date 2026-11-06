"""Validation must survive `python -O` where asserts strip to no-ops."""
import subprocess
import sys
import tempfile
from pathlib import Path


def test_validate_destination_raises_under_python_O():
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "out-Tau"
        dest.mkdir()
        code = ("import sys; sys.path.insert(0, %r);"
                "from pipeline.surgery.checkpoint.destination import validate_destination;"
                "validate_destination(%r)") % (str(root), str(dest))
        result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True)
        assert result.returncode != 0                        # FileExistsError still raised
        assert b"FileExistsError" in result.stderr
