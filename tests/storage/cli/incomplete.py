"""Store with manifest but no complete marker: CLI inspect exits nonzero."""
import subprocess
import sys
import tempfile
from pathlib import Path
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_cli_inspect_nonzero_on_incomplete_store():
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "w.safetensors", [("a", "F32", [1], b"\x01" * 4)])
        store = Path(tmp) / "store-Tau"
        split_checkpoint(src, store)
        (store / "complete.toml").unlink()
        result = subprocess.run(
            [sys.executable, "-B", str(root / "pipeline/cli/ms.py"), "weights",
             "inspect", str(store), "--verify"], capture_output=True, text=True)
        assert result.returncode != 0
        assert "complete: False" in result.stdout
