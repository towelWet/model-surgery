"""The offline case command must load its shipped wire schemas in a fresh process."""
import subprocess
import sys
from pathlib import Path


def test_offline_cases_loads_packaged_dependencies():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, "-B", str(root / "pipeline/cli/ms.py"), "cases"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "evaluation_fingerprint = " in result.stdout
    assert "0 errors" in result.stdout
