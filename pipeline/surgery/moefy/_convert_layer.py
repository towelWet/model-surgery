# _convert_layer.py — cluster, fit router, measure and pack one dense MLP layer.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.surgery.moefy._load_calib import load_calib
from pipeline.surgery.moefy._neuron_acts import neuron_acts
from pipeline.surgery.moefy.balanced._kmeans import balanced_kmeans
from pipeline.surgery.moefy.expert._norms import expert_norms
from pipeline.surgery.moefy._fit_router import fit_router
from pipeline.surgery.moefy._route_error import route_error
from pipeline.surgery.moefy._layer_tensors import layer_tensors
from pipeline.surgery.moefy._dense_f32 import dense_f32


def convert_layer(w, li, train, held, plan):
    """plan keys: experts, topk, logit_std, group, bits, block. Returns (tensors, report row)."""
    p = f"model.layers.{li}.mlp."
    wg, wu, wd = (dense_f32(w, p + n + ".weight") for n in ("gate_proj", "up_proj", "down_proj"))
    x = load_calib(train, li)
    acts = neuron_acts(x, wg, wu)
    labels = balanced_kmeans(acts, plan["experts"])
    router = fit_router(x, expert_norms(acts, wd, labels, plan["experts"]))
    router *= plan["logit_std"] / (x @ router).std()          # tiny logits: renormalised top-K weights ~ 1/K
    xh = load_calib(held, li)                                 # never used for fitting
    err = route_error(neuron_acts(xh, wg, wu), wd, labels, xh @ router, plan["topk"], plan["experts"])
    row = {"layer": li, "train_tokens": len(x), "heldout_tokens": len(xh), **err}
    return layer_tensors(li, wg, wu, wd, labels, router, plan["topk"], plan["group"], plan["bits"], block=plan["block"]), row
