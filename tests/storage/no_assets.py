"""Zero-asset source (shard only, no config/index/tokenizer) splits, validates, joins."""
import tempfile
from pathlib import Path
from pipeline.storage.inspection import inspect_store
from pipeline.storage.manifest.read import read_manifest
from pipeline.storage.manifest.validate import validate_manifest
from pipeline.storage.reassemble import reassemble_checkpoint
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_zero_asset_source_round_trips_byte_identical():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "w.safetensors", [("w.f32", "F32", [2], b"\x01\x02\x03\x04\x05\x06\x07\x08")])
        store = Path(tmp) / "store-Tau"
        split_checkpoint(src, store)
        assert "assets = []" in (store / "manifest.toml").read_text()
        plan = read_manifest(store)                            # parses + enforces schema/kind
        assert plan["assets"] == []                            # empty array surfaced, not omitted
        validate_manifest(plan, store)                         # missing/empty assets must not crash
        assert inspect_store(store)["complete"] is True        # store reports join-ready
        out = Path(tmp) / "out-Tau"
        reassemble_checkpoint(store, out)
        assert (out / "w.safetensors").read_bytes() == (src / "w.safetensors").read_bytes()
