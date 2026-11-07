"""Full projection writer round-trips packed weights and preserves nested assets."""
import sys
import tempfile
from pathlib import Path
import mlx.core as mx
from tests.checkpoint.helpers.mlx import write_mlx_checkpoint
from pipeline.surgery.shard._load_index import load_index
from pipeline.surgery.checkpoint.read_weight import read_weight


def test_projection_stream_preserves_quantization_and_assets():
    folder = Path(__file__).resolve().parents[2] / "pipeline/surgery/abliterate_parts"
    sys.path.insert(0, str(folder))
    from _run_abliterate import run_abliterate
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        weight = mx.arange(192, dtype=mx.float32).reshape(3, 64) / 192
        q, scales, biases = mx.quantize(weight, bits=4, group_size=32)
        key = "decoder.blocks.0.mlp.down_proj.weight"
        prefix = key.removesuffix(".weight")
        cfg = {"quantization": {"bits": 4, "group_size": 32}}
        src = write_mlx_checkpoint(root / "src", {key: q, prefix + ".scales": scales,
                                   prefix + ".biases": biases}, cfg)
        (src / "vision").mkdir()
        (src / "vision/processor.bin").write_bytes(b"preserve")
        dest = root / "projected"
        run_abliterate(src, dest, mx.array([1.0, 0, 0]), 1, cfg)
        dense, options, _ = read_weight(dest, key, load_index(dest)["weight_map"], {}, cfg)
        assert options == {"bits": 4, "group_size": 32}
        assert float(mx.max(mx.abs(dense[0])).item()) < 1e-6
        assert (dest / "vision/processor.bin").read_bytes() == b"preserve"
