"""Expert pruning slices routed tensors per layer, leaves shared experts, refuses unsafe counts."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import tempfile
from pathlib import Path
import numpy as np
from pipeline.gguf.format.read.gguf import read_gguf
from pipeline.gguf.format.write.gguf import write_gguf
from pipeline.gguf.edit.drop.experts import drop_experts
from pipeline.gguf.edit.kv_entry import kv_entry
from tests.gguf.helpers.moe_fixture import moe_fixture
from tests.gguf.helpers.read_tensor import read_tensor


def test_per_layer_plan_keeps_exact_expert_slices():
    plan = {0: [0, 1, 3], 1: [1, 2, 3], 2: [0, 2, 3]}
    with tempfile.TemporaryDirectory() as tmp:
        src, out = moe_fixture(str(Path(tmp) / "s.gguf")), str(Path(tmp) / "o.gguf")
        model = read_gguf(src)
        assert drop_experts(model, plan.__getitem__) == 3
        write_gguf(src, out, model)
        assert kv_entry(read_gguf(out), "fixmoe.expert_count")["value"] == 3
        for layer, keep in plan.items():
            for name in ("ffn_up_exps", "ffn_gate_inp"):
                full = read_tensor(src, f"blk.{layer}.{name}.weight")
                assert np.array_equal(read_tensor(out, f"blk.{layer}.{name}.weight"), full[keep])
            shexp = f"blk.{layer}.ffn_up_shexp.weight"
            assert np.array_equal(read_tensor(out, shexp), read_tensor(src, shexp))
        for bad in ({0: [0], 1: [0], 2: [0]}, {0: [0, 1], 1: [0, 1, 2], 2: [0, 1]}, {0: [1, 0], 1: [0, 1], 2: [0, 1]}):
            try:
                drop_experts(read_gguf(src), bad.__getitem__)          # below used / uneven / unsorted
            except ValueError:
                continue
            raise AssertionError(f"accepted {bad}")
