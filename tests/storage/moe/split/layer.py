"""strategy "layer" groups attention + router + experts per layer; layer-less tensors go shared."""
import json
import tempfile
from pathlib import Path
from pipeline.storage.experts.split_experts import split_experts
from tests.storage.helpers.moe_fixture import moe_fixture


def test_moe_split_by_layer():
    with tempfile.TemporaryDirectory() as tmp:
        src = moe_fixture(tmp, stacked=True)
        split_experts(src, Path(tmp) / "store", "layer", 1)
        wm = json.loads((Path(tmp) / "store" / "model.safetensors.index.json").read_text())["weight_map"]
        assert wm["model.embed_tokens.weight"] == "model-shared.safetensors"
        assert wm["model.layers.0.mlp.gate.weight"] == "model-layer-000.safetensors"
        assert wm["model.layers.0.mlp.switch_mlp.up_proj.weight.e001"] == "model-layer-000.safetensors"
