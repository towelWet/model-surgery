"""Malformed headers rejected: dup keys, bool dims, overlap/gap, size mismatch, unknown dtype."""
import json
import tempfile
from pathlib import Path
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.raw import write_shard


def test_malformed_shard_headers_all_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "store-Tau"
        cases = {
            "dup": b'{"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},'
                   b' "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}',
            "booldim": json.dumps({"a": {"dtype": "F32", "shape": [True],
                                         "data_offsets": [0, 4]}}).encode(),
            "gap": json.dumps({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                               "b": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]}}).encode(),
            "size": json.dumps({"a": {"dtype": "F32", "shape": [2],
                                      "data_offsets": [0, 4]}}).encode(),
            "dtype": json.dumps({"a": {"dtype": "FP9", "shape": [1],
                                       "data_offsets": [0, 4]}}).encode(),
        }
        for label, table in cases.items():
            src = Path(tmp) / f"src-{label}"
            src.mkdir()
            write_shard(src / "w.safetensors", table, b"\x00" * 8)
            try:
                split_checkpoint(src, dest)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted malformed header: {label}")
            assert not dest.exists()
