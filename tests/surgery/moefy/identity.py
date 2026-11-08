"""With top-K = all experts, converted MoE layer reproduces dense MLP (up to 8-bit quantization)."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx


def test_all_experts_reproduce_dense_mlp():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.moefy._layer_tensors import layer_tensors
    from pipeline.surgery.moefy._neuron_acts import neuron_acts
    rng = np.random.default_rng(4)
    h, inter, e = 64, 256, 4                                 # expert width 64 = one quant group
    wg, wu, wd = rng.normal(size=(inter, h)), rng.normal(size=(inter, h)), rng.normal(size=(h, inter))
    labels = rng.permutation(np.repeat(np.arange(e), inter // e))  # scattered neurons per expert
    t = layer_tensors(0, wg, wu, wd, labels, rng.normal(size=(h, e)) * 1e-3, e, 64, 8)
    g, u, d = (np.array(mx.dequantize(*(t[f"model.layers.0.mlp.switch_mlp.{n}.{s}"] for s in ("weight", "scales", "biases")),
                                      group_size=64, bits=8).astype(mx.float32)) for n in ("gate_proj", "up_proj", "down_proj"))
    x = rng.normal(size=(10, h))
    moe = sum(neuron_acts(x, g[k], u[k]) @ d[k].T / e for k in range(e))  # runtime weight ~1/K each
    dense = neuron_acts(x, wg, wu) @ wd.T
    assert np.linalg.norm(moe - dense) / np.linalg.norm(dense) < 0.03
