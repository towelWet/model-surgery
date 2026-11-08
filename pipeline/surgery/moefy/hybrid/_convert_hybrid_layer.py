# ../moefy/hybrid/_convert_hybrid_layer.py — one dense Qwen3.5 MLP -> shared expert + E routed experts + fitted router.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Same clustering / router fit / held-out route error as moefy (Qwen3), applied to routed neurons
# only; shared neurons are always on and excluded from route error (they never miss).
from pipeline.surgery.moefy._load_calib import load_calib
from pipeline.surgery.moefy._neuron_acts import neuron_acts
from pipeline.surgery.moefy.balanced._kmeans import balanced_kmeans
from pipeline.surgery.moefy.expert._norms import expert_norms
from pipeline.surgery.moefy._fit_router import fit_router
from pipeline.surgery.moefy._route_error import route_error
from pipeline.surgery.moefy._layer_tensors import layer_tensors
from pipeline.surgery.moefy._dense_f32 import dense_f32
from pipeline.surgery.moefy.hybrid.shared._split import shared_split
from pipeline.surgery.moefy.hybrid.shared._tensors import shared_tensors


def convert_hybrid_layer(w, pre, li, train, held, plan):
    """plan keys: experts, topk, shared, logit_std, group, bits. Returns (tensors, report row)."""
    p = f"{pre}{li}.mlp."
    wg, wu, wd = (dense_f32(w, p + n + ".weight") for n in ("gate_proj", "up_proj", "down_proj"))
    x = load_calib(train, li)
    acts = neuron_acts(x, wg, wu)
    sh, rt = shared_split(acts, wd, plan["shared"])
    labels = balanced_kmeans(acts[:, rt], plan["experts"])
    router = fit_router(x, expert_norms(acts[:, rt], wd[:, rt], labels, plan["experts"]))
    router *= plan["logit_std"] / (x @ router).std()          # tiny logits: renormalised top-K weights ~ 1/K
    xh = load_calib(held, li)                                 # never used for fitting
    err = route_error(neuron_acts(xh, wg[rt], wu[rt]), wd[:, rt], labels, xh @ router, plan["topk"], plan["experts"])
    out = layer_tensors(li, wg[rt], wu[rt], wd[:, rt], labels, router, plan["topk"], plan["group"], plan["bits"], pre)
    out.update(shared_tensors(p, wg, wu, wd, sh, plan["group"], plan["bits"]))
    return out, {"layer": li, "train_tokens": len(x), "heldout_tokens": len(xh), **err}
