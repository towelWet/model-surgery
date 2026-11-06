"""Tensor sampling must derive names from data and cover sparse, non-Qwen geometry."""
from pipeline.profile.verify.similarity.selection import select_tensors


def test_similarity_selection_is_discovered_and_bounded():
    table = {}
    for layer in (2, 7, 15, 90):
        for suffix in ("weight", "scales", "biases"):
            table[f"unfamiliar.layers.{layer}.matrix.{suffix}"] = (None, {"shape": [2, 2]}, 0)
    names = select_tensors(table, table, limit=2)
    assert names == ["unfamiliar.layers.15.matrix", "unfamiliar.layers.90.matrix"]
    assert len(select_tensors(table, table, limit=0)) == 4
    try:
        select_tensors(table, table, ["absent"])
    except ValueError:
        pass
    else:
        raise AssertionError("absent tensor accepted")
