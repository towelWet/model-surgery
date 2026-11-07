"""Actual neuron CLI edits packed 2/4/8-bit weights, with companions in separate shards."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import mlx.core as mx
from tests.checkpoint.helpers.mlx import write_mlx_checkpoint
from pipeline.surgery.shard._load_index import load_index
from pipeline.surgery.checkpoint.read_weight import read_weight


def test_neuron_cli_quantization_roundtrip():
    root = Path(__file__).resolve().parents[2]
    key = "decoder.blocks.0.mlp.down_proj.weight"
    for bits, group in ((2, 32), (4, 64), (8, 32)):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dense = mx.arange(4 * 128, dtype=mx.float32).reshape(4, 128) / 500
            q, scales, biases = mx.quantize(dense, bits=bits, group_size=group)
            prefix = key.removesuffix(".weight")
            src = write_mlx_checkpoint(base / "src", {key: q, prefix + ".scales": scales,
                prefix + ".biases": biases}, {"quantization": {"bits": bits, "group_size": group}})
            spec = base / "spec.json"
            spec.write_text(json.dumps({"layer": 0, "proj": "down_proj.weight", "columns": [7],
                                        "direction": [1, -1, 1, -1], "alpha": 0.5}))
            dest = base / "edited"
            result = subprocess.run([sys.executable, "-B", str(root / "pipeline/surgery/neuron/edit.py"),
                str(src), str(dest), "--spec", str(spec)], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            cfg = json.loads((dest / "config.json").read_text())
            actual, options, _ = read_weight(dest, key, load_index(dest)["weight_map"], {}, cfg)
            expected = mx.dequantize(q, scales, biases, bits=bits, group_size=group)
            expected = expected.at[:, 7].add(mx.array([0.5, -0.5, 0.5, -0.5]))
            assert options == {"bits": bits, "group_size": group}
            assert float(mx.max(mx.abs(actual - expected)).item()) < 0.22
