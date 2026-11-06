"""Patched write dies after one file: incomplete store, recover lists missing, resume heals."""
import contextlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch
from pipeline.storage.commands.recover.manifest import run as recover_run
from pipeline.storage.partition import materialize
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st
from tests.storage.helpers.flaky import flaky_write
from tests.storage.helpers.mutate import STATE


def test_write_interruption_incomplete_recover_then_resume():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "w.safetensors", [("a", "F32", [1], b"\x01" * 4)])
        (src / "config.json").write_text("{}")
        store = Path(tmp) / "store-Tau"
        STATE["real"], STATE["calls"] = materialize.write_xb, 0

        with patch.object(materialize, "write_xb", flaky_write):
            try:
                split_checkpoint(src, store)
            except OSError:
                pass
            else:
                raise AssertionError("interruption did not propagate")
        assert not (store / "complete.toml").exists()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = recover_run([str(store)])                # weights recover, in-process
        assert code == 1 and "missing" in out.getvalue()
        split_checkpoint(src, store, resume=True)
        assert (store / "complete.toml").exists()
        baseline = Path(tmp) / "baseline-Tau"
        split_checkpoint(src, baseline)
        got = {p.relative_to(store): p for p in store.rglob("*") if p.is_file()}
        want = {p.relative_to(baseline): p for p in baseline.rglob("*") if p.is_file()}
        assert set(got) == set(want)                        # same file set as clean split
        for rel, path in got.items():
            assert path.read_bytes() == want[rel].read_bytes(), rel
