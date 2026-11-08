"""Actual DARE CLI respects mixed-bit source metadata and emits honest method labels."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import mlx.core as mx
from tests.checkpoint.helpers.mlx import write_mlx_checkpoint
from pipeline.surgery.shard.load._index import load_index
from pipeline.surgery.checkpoint.read_weight import read_weight


def test_merge_cli_preserves_base_quantization_and_dense_export_metadata():
    root = Path(__file__).resolve().parents[2]
    key = "decoder.blocks.0.mlp.down_proj.weight"
    prefix = key.removesuffix(".weight")
    with tempfile.TemporaryDirectory() as tmp:
        folder, decoded, sources = Path(tmp), [], []
        for i, (bits, group) in enumerate(((2, 32), (8, 64))):
            weight = mx.arange(128, dtype=mx.float32).reshape(2, 64) / 128 + i
            q, s, b = mx.quantize(weight, bits=bits, group_size=group)
            cfg = {"model_type": "fixture", "quantization": {"bits": 4, "group_size": 64,
                   prefix: {"bits": bits, "group_size": group}}}
            sources.append(write_mlx_checkpoint(folder / f"src-{i}",
                {key: q, prefix + ".scales": s, prefix + ".biases": b}, cfg))
            decoded.append(mx.dequantize(q, s, b, bits=bits, group_size=group))
        for dense in (False, True):
            dest = folder / ("dense" if dense else "quantized")
            args = [sys.executable, "-B", str(root / "pipeline/surgery/merge/ties/core.py"),
                    *map(str, sources), str(dest), "--density", "1", "--lambda", "0.5"]
            result = subprocess.run(args + (["--bf16-only"] if dense else []), capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            cfg = json.loads((dest / "config.json").read_text())
            index = load_index(dest)
            actual, options, _ = read_weight(dest, key, index["weight_map"], {}, cfg)
            expected = (decoded[0] + decoded[1]) / 2
            assert float(mx.max(mx.abs(actual - expected)).item()) < 0.18
            assert index["metadata"]["merge_method"] == "dare_task_arithmetic"
            assert options == (None if dense else {"bits": 2, "group_size": 32})
            assert ("quantization" not in cfg) == dense
