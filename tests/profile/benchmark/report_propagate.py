"""report.sh propagates preflight's exit status (stub preflight, temp fixture copy)."""
import shutil
import subprocess
import tempfile
from pathlib import Path

REPORT = Path(__file__).resolve().parents[3] / "pipeline" / "profile" / "benchmark" / "report.sh"


def test_report_propagates_preflight_status():
    with tempfile.TemporaryDirectory() as tmp:
        bench = Path(tmp) / "profile" / "benchmark"
        bench.mkdir(parents=True)
        shutil.copy(REPORT, bench / "report.sh")              # fixture copy, repo untouched
        preflight = Path(tmp) / "profile" / "preflight.sh"
        for status in (0, 1):
            preflight.write_text(f"#!/usr/bin/env bash\nexit {status}\n")
            result = subprocess.run(["bash", str(bench / "report.sh")], capture_output=True, timeout=10)
            assert result.returncode == status
