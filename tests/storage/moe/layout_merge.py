"""layouts/*.toml are hot-loaded: lists union across files, conflicting scalars refuse."""
import tempfile
from pathlib import Path
from pipeline.storage.experts.load.layout import load_layout


def test_moe_layout_merge_and_conflict():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.toml").write_text('stacked_components = ["switch_mlp"]\nshared_file = "model-s.safetensors"\n')
        (Path(tmp) / "b.toml").write_text('stacked_components = ["block_sparse_moe", "switch_mlp"]\n')
        lay = load_layout(tmp)
        assert lay["stacked_components"] == ["switch_mlp", "block_sparse_moe"]   # ordered, deduped union
        (Path(tmp) / "c.toml").write_text('shared_file = "other.safetensors"\n')
        try:
            load_layout(tmp)
            raise AssertionError("conflicting scalar accepted")
        except ValueError as e:
            assert "shared_file" in str(e)
