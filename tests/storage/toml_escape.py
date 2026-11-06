"""Tensor names with control chars TOML-escape in the manifest and round-trip back."""
import tempfile
import tomllib
from pathlib import Path
from pipeline.storage.manifest.read import read_manifest
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_control_char_tensor_name_round_trips_manifest():
    name = 'w\nq"t\\x\x01'                                  # newline, quote, backslash, C0
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "w.safetensors", [(name, "F32", [1], b"\x01" * 4)])
        store = Path(tmp) / "store-Tau"
        split_checkpoint(src, store)                        # escaping happens at write time
        plan = read_manifest(store)
        assert plan["tensors"][0]["name"] == name
        raw = (store / "manifest.toml").read_text()
        assert tomllib.loads(raw)["tensors"][0]["name"] == name
