"""moe-split destacks [E,...] into <name>.eNNN rows per expert file; moe-join restores identical bytes."""
import json
import tempfile
from pathlib import Path
from pipeline.storage.experts.join_experts import join_experts
from pipeline.storage.experts.read_tensors import read_tensors
from pipeline.storage.experts.split_experts import split_experts
from tests.storage.helpers.moe_fixture import moe_fixture

UP = "model.layers.0.mlp.switch_mlp.up_proj.weight"


def test_moe_split_stacked_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        src, store = moe_fixture(tmp, stacked=True), Path(tmp) / "store"
        r = split_experts(src, store, "expert", 2)
        assert set(r["files"]) == {"model-shared.safetensors", "model-experts-000.safetensors",
                                   "model-experts-001.safetensors"}       # model-* = tensor-array loadable
        wm = json.loads((store / "model.safetensors.index.json").read_text())["weight_map"]
        assert wm[UP + ".e003"] == "model-experts-001.safetensors"
        t = {x["name"]: x for x in read_tensors(store)}[UP + ".e002"]
        assert t["shape"] == [2]                                         # leading E dim dropped
        with open(t["path"], "rb") as f:
            f.seek(t["start"])
            assert f.read(t["nbytes"]) == bytes(range(216, 224))         # source row 2 exactly
        join_experts(store, Path(tmp) / "out")
        a, b = read_tensors(src), read_tensors(Path(tmp) / "out")
        assert [(x["name"], x["shape"]) for x in a] == [(x["name"], x["shape"]) for x in b]
        for x, y in zip(a, b):
            with open(x["path"], "rb") as fx, open(y["path"], "rb") as fy:
                fx.seek(x["start"]); fy.seek(y["start"])
                assert fx.read(x["nbytes"]) == fy.read(y["nbytes"]), x["name"]
        assert (Path(tmp) / "out" / "tokenizer.json").exists()
