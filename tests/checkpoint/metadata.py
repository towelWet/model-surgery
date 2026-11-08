"""Inspection works for single-file and indexed checkpoints without MLX imports."""
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from tests.checkpoint.helpers.fixture import write_checkpoint
from pipeline.checkpoint.summary import summarize


def test_checkpoint_inspection_cli_and_index():
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        folder = write_checkpoint(Path(tmp) / "model", config={"text_config": {
            "model_type": "fixture_moe", "num_hidden_layers": 1, "num_local_experts": 3}})
        summary = summarize(folder)
        assert summary["experts"] == 3 and summary["layer_ids"] == "2"
        index = folder / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"net.layers.2.ffn.weight": "weights.safetensors"}}))
        assert summarize(folder) == summary
        output = Path(tmp) / "summary.toml"
        command = [sys.executable, "-B", str(root / "pipeline/cli/ms.py"), "inspect", str(folder), "--out", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert tomllib.loads(output.read_text())["tensors"] == 1
        assert subprocess.run(command, capture_output=True).returncode == 1
