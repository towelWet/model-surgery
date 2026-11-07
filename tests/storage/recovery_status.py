"""Actual recovery CLI returns failure for an incomplete store even with no missing entries."""
import subprocess
import sys
import tempfile
from pathlib import Path


def test_recover_cli_incomplete_exit_status():
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "incomplete"
        store.mkdir()
        (store / "transaction.toml").write_text('operation = "split"\n')
        result = subprocess.run([sys.executable, "-B", str(root / "pipeline/cli/ms.py"),
                                 "weights", "recover", str(store)], capture_output=True, text=True)
        assert result.returncode != 0
        assert "incomplete store" in result.stdout
