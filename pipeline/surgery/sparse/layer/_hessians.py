# layer/_hessians.py — per-role Hessian H = 2XᵀX/N of one decoder layer's linear inputs (SparseGPT).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Inputs from dense-model capture (attn / o / mlp taps; down input recomputed from dense gate/up),
# same simplification as wquant: no re-feeding through already-pruned earlier layers.
import numpy as np
import mlx.core as mx
from pipeline.surgery.moefy._load_calib import load_calib
from pipeline.surgery.moefy._neuron_acts import neuron_acts


def layer_hessians(lazy, key, layer, files):
    tag = f"layers.{layer}."
    p = key[: key.index(tag) + len(tag)] + "mlp."             # same layer's MLP, whatever role `key` has
    wg, wu = (np.array(lazy[p + n + "_proj.weight"].astype(mx.float32)) for n in ("gate", "up"))
    xm = load_calib(files, layer)
    h = lambda x: 2.0 * (x.T.astype(np.float64) @ x) / len(x)
    ha = h(load_calib(files, layer, "attn"))
    hm = h(xm)
    return {"q": ha, "k": ha, "v": ha, "o": h(load_calib(files, layer, "o")),
            "gate": hm, "up": hm, "down": h(neuron_acts(xm, wg, wu))}
