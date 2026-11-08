"""dry_run plans fully but creates no destination and writes nothing."""
import tempfile
from pathlib import Path
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_dry_run_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "w.safetensors", [("a", "F32", [2], b"\x00" * 8)])
        dest = Path(tmp) / "store-Tau"
        plan = split_checkpoint(src, dest, dry_run=True)
        assert plan["dry_run"] and plan["source_fingerprint"]
        assert plan["required_bytes"] > 0
        assert not dest.exists()
        assert sorted(p.name for p in src.iterdir()) == ["w.safetensors"]
