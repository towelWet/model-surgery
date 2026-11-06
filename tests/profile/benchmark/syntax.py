"""Benchmark shell entrypoints parse cleanly under bash -n."""
import subprocess
from pathlib import Path

PROFILE = Path(__file__).resolve().parents[3] / "pipeline" / "profile"


def test_benchmark_scripts_bash_syntax_ok():
    for script in ("benchmark/run.sh", "benchmark/report.sh", "preflight.sh"):
        result = subprocess.run(["bash", "-n", str(PROFILE / script)], capture_output=True)
        assert result.returncode == 0, f"{script}: {result.stderr.decode()}"
