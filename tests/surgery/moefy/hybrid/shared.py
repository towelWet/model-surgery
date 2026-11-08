"""Shared split: top-contribution neurons, disjoint complete partition; shared gate = 0 with doubled down."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
import mlx.core as mx


def test_shared_split_and_gate_compensation():
    from pipeline.surgery.moefy.hybrid.shared._split import shared_split
    from pipeline.surgery.moefy.hybrid.shared._tensors import shared_tensors
    rng = np.random.default_rng(1)
    acts, wd = rng.normal(size=(40, 8)), rng.normal(size=(4, 8))
    acts[:, 5] *= 50.0                                       # neuron 5 dominates output
    sh, rt = shared_split(acts, wd, 2)
    assert 5 in sh and len(sh) == 2 and sorted(np.concatenate([sh, rt]).tolist()) == list(range(8))
    wg, wu, wdd = (rng.normal(size=(64, 64)).astype(np.float32) for _ in range(3))
    t = shared_tensors("L.mlp.", wg, wu, wdd, np.arange(64), 64, 8)
    g = t["L.mlp.shared_expert_gate.weight"]
    assert g.shape == (1, 64) and mx.all(g == 0).item()      # sigmoid(0) = 0.5 for every token
    d = mx.dequantize(*(t["L.mlp.shared_expert.down_proj." + s] for s in ("weight", "scales", "biases")), group_size=64, bits=8)
    assert np.allclose(np.array(d.astype(mx.float32)), wdd * 2.0, atol=0.1)     # doubled to undo 0.5 gate
