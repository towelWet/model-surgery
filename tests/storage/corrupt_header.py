"""Resume refuses to overwrite a same-length corrupted stored raw header."""
import tempfile
from pathlib import Path
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_resume_refuses_corrupted_raw_header_no_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "w.safetensors", [("a", "F32", [1], b"\x01" * 4)])
        store = Path(tmp) / "store-Tau"
        split_checkpoint(src, store)
        (store / "complete.toml").unlink()
        hfile = next((store / "headers").glob("*.bin"))
        damaged = bytes(b ^ 0xFF for b in hfile.read_bytes())   # same length, wrong bytes
        hfile.write_bytes(damaged)
        try:
            split_checkpoint(src, store, resume=True)
        except ValueError:
            pass
        else:
            raise AssertionError("resume overwrote corrupted raw header")
        assert hfile.read_bytes() == damaged                    # bytes left untouched
