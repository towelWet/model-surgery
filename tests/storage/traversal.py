"""Index entries or shard paths escaping the source dir are rejected before writes."""
import json
import tempfile
from pathlib import Path
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_traversal_rejected_before_destination():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "ok.safetensors", [("a", "F32", [1], b"\x00" * 4)])
        (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "a": "ok.safetensors", "b": "../escape.safetensors"}, "metadata": {}}))
        dest = Path(tmp) / "store-Tau"
        try:
            split_checkpoint(src, dest)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted escaping shard path")
        assert not dest.exists()
