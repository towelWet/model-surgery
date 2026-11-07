"""Default writer packs many small groups into few shards instead of one file per tensor."""
import tempfile
from pathlib import Path
import mlx.core as mx
from tests.checkpoint.helpers.mlx import write_mlx_checkpoint
from pipeline.surgery.shard._load_index import load_index
from pipeline.surgery.checkpoint.tensor_groups import tensor_groups
from pipeline.surgery.checkpoint.write_groups import write_groups


def test_default_budget_packs_groups_and_small_budget_splits_them():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        weights = {f"decoder.blocks.{i}.mlp.up_proj.weight": mx.ones((2, 32)) * i for i in range(8)}
        source = write_mlx_checkpoint(base / "source", weights, {"model_type": "fixture"})
        mapping = load_index(source)["weight_map"]
        for budget, shards in ((None, 1), (512, 4)):              # 256 B per tensor: 2 tensors per 512 B shard
            dest = base / f"out-{shards}"
            extra = {} if budget is None else {"shard_bytes": budget}
            write_groups(source, dest, tensor_groups(source, mapping, {}), load_index(source), **extra)
            written = load_index(dest)["weight_map"]
            assert set(written) == set(weights) and len(set(written.values())) == shards
            for key, shard in written.items():
                assert mx.array_equal(mx.load(str(dest / shard))[key], weights[key]).item()
