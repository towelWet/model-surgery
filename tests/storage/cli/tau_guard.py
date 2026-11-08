"""python -O CLI accepts general destination names and rejects invalid source before writes."""
import subprocess
import sys
import tempfile
from pathlib import Path


def test_optimized_cli_general_name_reaches_source_validation():
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()                                      # no shards: plan would fail AFTER
        for extra in ([], ["--dry-run"]):
            dest = Path(tmp) / ("dry-store" if extra else "plain-store")
            result = subprocess.run(
                [sys.executable, "-O", "-B",
                 str(root / "pipeline/cli/ms.py"), "weights", "split",
                 str(src), str(dest), *extra],
                capture_output=True, text=True)
            assert result.returncode != 0
            assert not dest.exists()
            combined = result.stdout + result.stderr
            assert "no safetensors shards" in combined
            assert "destination name must include" not in combined
