"""TIES CLI merges three real MLX fixtures end to end and labels method honestly."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import subprocess
import sys
import tempfile
from pathlib import Path
import mlx.core as mx
import numpy as np
from tests.checkpoint.helpers.mlx import write_mlx_checkpoint
from pipeline.surgery.shard.load._index import load_index
from pipeline.surgery.checkpoint.read_weight import read_weight


def test_ties_cli_three_sources_and_refuses_single_finetune():
    root = Path(__file__).resolve().parents[3]
    key = "decoder.blocks.0.mlp.down_proj.weight"
    cfg = {"model_type": "fixture"}
    rows = [[0, 0, 0, 0], [3, -1, 0.1, 2], [1, -4, -0.2, -1]]  # base + two task vectors (hand example)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        src = [write_mlx_checkpoint(folder / f"s{i}", {key: mx.array([r], dtype=mx.float32)}, cfg)
               for i, r in enumerate(rows)]
        cli = [sys.executable, "-B", str(root / "pipeline/surgery/merge/ties/core.py")]
        one = subprocess.run(cli + [str(src[0]), str(src[1]), str(folder / "x"), "--method", "ties"],
                             capture_output=True, text=True)
        assert one.returncode != 0 and not (folder / "x").exists()   # TIES needs two fine-tunes
        dest = folder / "out"
        run = subprocess.run(cli + [*map(str, src), str(dest), "--method", "ties", "--density", "0.75"],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        index = load_index(dest)
        got, _, _ = read_weight(dest, key, index["weight_map"], {}, cfg)
        assert np.allclose(np.array(got.astype(mx.float32)), [[2.0, -2.5, 0.0, 2.0]])
        assert index["metadata"]["merge_method"] == "ties"
        assert index["metadata"]["merge_others"] == "s1,s2"
