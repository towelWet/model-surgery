"""Lone model.safetensors with "__metadata__": null still yields dict metadata (slerp crash 2026-11-08)."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import struct
import tempfile
from pathlib import Path
from pipeline.surgery._index_or_single import synthesize_index


def test_null_header_metadata_becomes_empty_dict():
    header = {"__metadata__": None,                                   # what tensor-array fuse writes
              "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)   # u64 header length + header + one f32
        index = synthesize_index(tmp)
    assert index["metadata"] == {}                                    # dict(), so run_merge can dict() it
    assert index["weight_map"] == {"w": "model.safetensors"}
