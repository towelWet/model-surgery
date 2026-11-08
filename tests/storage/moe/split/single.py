"""moe-split moves HF per-expert tensors whole into their expert file; router + shared expert stay shared."""
import json
import tempfile
from pathlib import Path
from pipeline.storage.experts.split_experts import split_experts
from tests.storage.helpers.moe_fixture import moe_fixture


def test_moe_split_per_expert_index():
    with tempfile.TemporaryDirectory() as tmp:
        src = moe_fixture(tmp, stacked=False)
        r = split_experts(src, Path(tmp) / "store", "expert", 1)
        assert len(r["files"]) == 5                                      # shared + one file per expert
        wm = json.loads((Path(tmp) / "store" / "model.safetensors.index.json").read_text())["weight_map"]
        assert wm["model.layers.0.mlp.experts.3.up_proj.weight"] == "model-experts-003.safetensors"
        assert wm["model.layers.0.mlp.gate.weight"] == "model-shared.safetensors"            # router
        assert wm["model.layers.0.mlp.shared_expert.up_proj.weight"] == "model-shared.safetensors"
