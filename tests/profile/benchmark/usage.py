"""run.sh requires exactly one config filepath; wrong argc fails before any request."""
import subprocess
from pathlib import Path

RUN = Path(__file__).resolve().parents[3] / "pipeline" / "profile" / "benchmark" / "run.sh"


def test_run_sh_rejects_wrong_arg_count():
    for argv in ([], ["a.toml", "b.toml"]):
        result = subprocess.run(["bash", str(RUN), *argv], capture_output=True, timeout=10)
        assert result.returncode != 0
        assert b"usage" in result.stderr.lower()
