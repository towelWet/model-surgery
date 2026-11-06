"""python -O CLI: non-Tau destination rejected before any source plan/hash; dry-run too."""
import subprocess
import sys
import tempfile
from pathlib import Path


def test_optimized_cli_tau_guard_precedes_source_plan():
    root = Path(__file__).resolve().parents[2]
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
            assert "Tau" in combined                     # destination error, not shard error
            assert "no safetensors shards" not in combined
