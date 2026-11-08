"""Truncated headers, invalid tensor ranges, and wrong indexes must not pass inspection."""
import json
import struct
import tempfile
from pathlib import Path
from tests.checkpoint.helpers.fixture import write_checkpoint
from pipeline.checkpoint.tensors import tensor_index


def test_checkpoint_corruption_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        for label, content in (("prefix", b"bad"), ("oversized", struct.pack("<Q", 2**63))):
            folder = write_checkpoint(Path(tmp) / label)
            (folder / "weights.safetensors").write_bytes(content)
            try:
                tensor_index(folder)
            except ValueError:
                pass
            else:
                raise AssertionError(label)
        folder = write_checkpoint(Path(tmp) / "index")
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"absent": "weights.safetensors"}}))
        try:
            tensor_index(folder)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong index accepted")
