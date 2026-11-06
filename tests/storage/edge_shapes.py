"""Scalar (shape []) and zero-dimension (shape [0]) tensors split+join byte-identical."""
import tempfile
from pathlib import Path
from pipeline.storage.reassemble import reassemble_checkpoint
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_scalar_and_zero_dim_tensors_round_trip_bytes():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "w.safetensors", [
            ("scalar", "F32", [], b"\x01\x02\x03\x04"),     # count=1 scalar
            ("empty", "F32", [0], b""),                     # count=0 zero-dim
            ("vec", "U8", [3], b"abc")])
        (src / "config.json").write_text("{}")              # zero-asset stores unjoinable
        store = Path(tmp) / "store-Tau"
        split_checkpoint(src, store)
        out = Path(tmp) / "out-Tau"
        reassemble_checkpoint(store, out)
        original = (src / "w.safetensors").read_bytes()
        assert (out / "w.safetensors").read_bytes() == original
