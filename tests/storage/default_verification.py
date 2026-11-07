"""Default inspection detects missing/corrupt files; cheap inventory never claims completion."""
import tempfile
from pathlib import Path
from pipeline.storage.inspection import inspect_store
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_default_inspect_cannot_claim_unverified_completion():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "source"
        src.mkdir()
        write_st(src / "w.safetensors", [("w", "F32", [1], b"\x00" * 4)])
        (src / "processor.bin").write_bytes(b"original")
        store = Path(tmp) / "store"
        split_checkpoint(src, store)
        assert inspect_store(store)["complete"]
        assert not inspect_store(store, verify=False)["complete"]
        (store / "assets/processor.bin").write_bytes(b"modified")
        report = inspect_store(store)
        assert not report["complete"] and "assets/processor.bin" in report["corrupt"]
        (store / "assets/processor.bin").unlink()
        report = inspect_store(store)
        assert not report["complete"] and "assets/processor.bin" in report["missing"]
