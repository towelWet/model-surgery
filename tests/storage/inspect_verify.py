"""inspect reports inventory without tensor loads; verify catches corrupted payloads."""
import tempfile
from pathlib import Path
from pipeline.storage.inspection import inspect_store
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_inspect_inventory_and_corruption_detection():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "w.safetensors", [("a", "F32", [2], b"\x01" * 8)])
        (src / "config.json").write_text("{}")
        store = Path(tmp) / "store-Tau"
        plan = split_checkpoint(src, store)
        report = inspect_store(store, verify=True)
        assert report["complete"] and report["tensors"] == 1 and report["corrupt"] == []
        assert report["payload_bytes"] == 8
        tfile = next((store / "tensors").glob("*.safetensors"))
        blob = bytearray(tfile.read_bytes())
        blob[-1] ^= 0xFF                                          # corrupt payload
        tfile.write_bytes(bytes(blob))
        report = inspect_store(store, verify=True)
        assert f"tensors/{tfile.name}" in report["corrupt"]
        assert report["complete"] is False
