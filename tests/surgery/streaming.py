"""Source cache is bounded and the shared output writer consumes groups lazily."""
import tempfile
from pathlib import Path
import mlx.core as mx
from tests.checkpoint.helpers.mlx import write_mlx_checkpoint
from pipeline.surgery.shard._load_cache import load_shard_cache
from pipeline.surgery.shard._load_index import load_index
from pipeline.surgery.checkpoint.tensor_groups import tensor_groups
from pipeline.surgery.checkpoint.write_groups import write_groups


def test_shard_cache_and_output_writer_do_not_accumulate_the_model():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        weights = {f"decoder.blocks.{i}.mlp.up_proj.weight": mx.ones((2, 32)) * i for i in range(8)}
        source = write_mlx_checkpoint(base / "source", weights, {"model_type": "fixture"})
        index = load_index(source)
        cache = {}
        for shard in index["weight_map"].values():
            load_shard_cache(source, shard, cache)
            assert len(cache) == 1
        groups = tensor_groups(source, index["weight_map"], cache)
        assert iter(groups) is groups
        destination = base / "output"
        write_groups(source, destination, groups, index, shard_bytes=1)   # flush after every group
        output = load_index(destination)
        assert set(output["weight_map"]) == set(weights)
        assert len(set(output["weight_map"].values())) == len(weights)
        assert not (destination / ".surgery-incomplete.toml").exists()
