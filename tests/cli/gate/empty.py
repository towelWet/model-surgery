"""An empty rules file used to promote arbitrary folders with no evidence."""
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def test_empty_and_report_only_rules_block_promotion():
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        rules = folder / "rules.toml"
        for source in ("", "[kl]\nmean_kl = { report_only = true }\n"):
            rules.write_text(source)
            result = subprocess.run([sys.executable, "-B", str(root / "pipeline/cli/ms.py"), "gate",
                                     "--base", tmp, "--cand", tmp, "--rules", str(rules)],
                                    capture_output=True, text=True)
            assert result.returncode == 1, result.stdout + result.stderr
            assert not tomllib.loads((folder / "verdict.toml").read_text())["passed"]
