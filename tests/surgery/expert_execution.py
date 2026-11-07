"""Actual expert CLI rejects incomplete coverage, then prunes both layers with exact row preservation."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import mlx.core as mx
from tests.checkpoint.helpers.mlx import write_mlx_checkpoint
from pipeline.surgery.shard._load_index import load_index
from pipeline.surgery.shard._load_cache import load_shard_cache


def test_expert_cli_coverage_and_streamed_result():
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        folder, tensors = Path(tmp), {}
        for layer in range(2):
            prefix = f"model.layers.{layer}.mlp."
            tensors[prefix + "gate.weight"] = mx.arange(8, dtype=mx.float32).reshape(4, 2)
            for expert in range(4):
                tensors[prefix + f"experts.{expert}.down_proj.weight"] = mx.ones((2, 2)) * expert
        source = write_mlx_checkpoint(folder / "source", tensors,
            {"num_experts": 4, "num_experts_per_tok": 2, "num_hidden_layers": 2})
        usage = folder / "usage.json"
        destination = folder / "pruned"
        command = [sys.executable, "-B", str(root / "pipeline/surgery/expert/prune.py"),
                   str(source), str(destination), str(usage), "--remove-frac", "0.5"]
        usage.write_text(json.dumps({"per_layer": {"0": {"prune_order": [3, 1, 2, 0]}}}))
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0 and not destination.exists()
        usage.write_text(json.dumps({"per_layer": {str(i): {"prune_order": [3, 1, 2, 0]} for i in range(2)}}))
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        index = load_index(destination)["weight_map"]
        for layer in range(2):
            key = f"model.layers.{layer}.mlp.gate.weight"
            actual = load_shard_cache(destination, index[key], {})[key]
            assert bool(mx.array_equal(actual, tensors[key][mx.array([0, 2])]).item())
            key = f"model.layers.{layer}.mlp.experts.1.down_proj.weight"
            actual = load_shard_cache(destination, index[key], {})[key]
            assert bool(mx.all(actual == 2).item())
