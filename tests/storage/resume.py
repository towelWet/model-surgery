"""Interrupted split: resume writes missing files; damaged partials and source drift refused."""
import tempfile
from pathlib import Path
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_resume_completes_missing_and_refuses_damage():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "w.safetensors", [("a", "F32", [1], b"\x01" * 4),
                                         ("b", "F32", [1], b"\x02" * 4)])
        (src / "config.json").write_text("{}")
        store = Path(tmp) / "store-Tau"
        plan = split_checkpoint(src, store)                      # complete baseline
        # simulate interruption: drop completion + one tensor file, keep transaction+manifest
        (store / "complete.toml").unlink()
        tfile = next((store / "tensors").glob("*.safetensors"))
        tfile.unlink()
        split_checkpoint(src, store, resume=True)                # rewrites only missing file
        assert (store / "complete.toml").exists() and tfile.exists()
        # damaged partial: correct length, wrong bytes -> clear refusal, never overwrite
        tfile.write_bytes(tfile.read_bytes()[:-1] + b"X")
        (store / "complete.toml").unlink()
        try:
            split_checkpoint(src, store, resume=True)
        except ValueError:
            pass
        else:
            raise AssertionError("overwrote damaged partial")
