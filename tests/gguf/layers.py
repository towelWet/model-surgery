"""Layer pruning drops blocks, renumbers survivors, shrinks per-layer arrays, keeps bytes."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import tempfile
from pathlib import Path
import numpy as np
from pipeline.gguf.format.read.gguf import read_gguf
from pipeline.gguf.format.write.gguf import write_gguf
from pipeline.gguf.edit.drop.layers import drop_layers
from pipeline.gguf.edit.kv_entry import kv_entry
from tests.gguf.helpers.moe_fixture import moe_fixture
from tests.gguf.helpers.read_tensor import read_tensor


def test_prune_middle_layer_renumbers_and_preserves_values():
    with tempfile.TemporaryDirectory() as tmp:
        src, out = moe_fixture(str(Path(tmp) / "s.gguf")), str(Path(tmp) / "o.gguf")
        model = read_gguf(src)
        assert drop_layers(model, [1]) == [0, 2]
        write_gguf(src, out, model)
        got = read_gguf(out)
        assert kv_entry(got, "fixmoe.block_count")["value"] == 2
        assert kv_entry(got, "fixmoe.attention.head_count")["value"]["items"] == [10, 12]
        assert not any(t["name"].startswith("blk.2.") for t in got["tensors"])
        assert np.all(read_tensor(out, "blk.1.attn_q.weight") == 2000)   # old layer 2 now layer 1
        assert np.array_equal(read_tensor(out, "token_embd.weight"), read_tensor(src, "token_embd.weight"))
        for bad in ([], [3], [0, 1, 2, 7]):
            try:
                drop_layers(read_gguf(src), bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted {bad}")
