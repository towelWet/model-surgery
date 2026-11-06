"""Real repack entry under python -O: existing dest sentinel survives, bad source writes nothing."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def test_repack_entry_guards_under_python_O():
    root = Path(__file__).resolve().parents[3]
    script = root / "pipeline/export/repack/1bit_to_2bit.py"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        (src / "config.json").write_text(json.dumps({"quantization": {"bits": 2}}))
        dest = Path(tmp) / "out-Tau"
        base = [sys.executable, "-O", "-B", str(script), str(src), str(dest)]
        result = subprocess.run(base, capture_output=True)          # non-1bit source
        assert result.returncode != 0 and not dest.exists()         # ValueError pre-mkdir
        dest.mkdir()
        (dest / "sentinel").write_bytes(b"keep")
        result = subprocess.run(base, capture_output=True)          # existing dest
        assert result.returncode != 0
        assert (dest / "sentinel").read_bytes() == b"keep"          # sentinel intact
