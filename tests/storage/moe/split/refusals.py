"""moe-split refuses dense configs; moe-join refuses tampered slices (hash check, no silent output)."""
import json
import tempfile
from pathlib import Path
from pipeline.storage.experts.join_experts import join_experts
from pipeline.storage.experts.split_experts import split_experts
from tests.storage.helpers.moe_fixture import moe_fixture


def test_moe_split_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        src = moe_fixture(tmp, stacked=True)
        store = Path(tmp) / "store"
        split_experts(src, store, "expert", 2)
        f = store / "model-experts-000.safetensors"
        raw = bytearray(f.read_bytes()); raw[-1] ^= 0xFF; f.write_bytes(bytes(raw))   # flip one payload byte
        try:
            join_experts(store, Path(tmp) / "out")
            raise AssertionError("tampered join accepted")
        except ValueError as e:
            assert "mismatch" in str(e)
        (src / "config.json").write_text(json.dumps({"hidden_size": 8}))             # dense: no expert count
        try:
            split_experts(src, Path(tmp) / "store2")
            raise AssertionError("dense split accepted")
        except ValueError as e:
            assert "no routed experts" in str(e)
