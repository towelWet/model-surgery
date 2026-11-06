"""Exercise real CLI pruning on a tiny synthetic checkpoint, including python -O."""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from tests.checkpoint.helpers.fixture import write_checkpoint
from pipeline.checkpoint.tensors import tensor_index


def test_depth_cli_preserves_survivor_bytes():
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        tensors = {f"model.layers.{i}.ffn.weight": {
            "dtype": "F32", "shape": [1, 1], "data_offsets": [i * 4, (i + 1) * 4]} for i in range(3)}
        parent = write_checkpoint(folder / "parent", tensors, {"num_hidden_layers": 3}, struct.pack("<fff", 11, 22, 33))
        (parent / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "weights.safetensors" for key in tensors}, "metadata": {}}))
        child = folder / "child-Tau"                            # dest names must include Tau
        command = [sys.executable, "-B", "-O", str(root / "pipeline/surgery/depth/prune.py"), str(parent), str(child), "1"]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads((child / "config.json").read_text())["num_hidden_layers"] == 2
        index = tensor_index(child)
        for key, expected in (("model.layers.0.ffn.weight", 11), ("model.layers.1.ffn.weight", 33)):
            path, entry, start = index[key]
            with path.open("rb") as stream:
                stream.seek(start + entry["data_offsets"][0])
                assert struct.unpack("<f", stream.read(4))[0] == expected
        assert subprocess.run(command, capture_output=True).returncode != 0
