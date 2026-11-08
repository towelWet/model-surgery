# ../moe/combine/_router_rows.py — router gate rows from each expert's own prompts (mergekit-moe "hidden" idea).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Row e = unit(mean MLP input on expert e's prompts − mean over all experts' prompts). Centering
# alone is not enough: with 2 experts rows are opposite, so shared input mean adds +c to one logit
# and -c to other. Rows are therefore also made orthogonal to global mean direction. Rows are then
# scaled so router logits on calibration rows have std `logit_std` (softmax neither flat nor one-hot).
import numpy as np
from pipeline.surgery.moefy._load_calib import load_calib


def router_rows(calib_sets, layer, logit_std=2.0):
    xs = [load_calib(files, layer) for files in calib_sets]   # per expert: [N_e, H] normed MLP inputs
    means = np.stack([x.mean(0) for x in xs])
    mu = means.mean(0)
    rows = means - mu
    u = mu / (np.linalg.norm(mu) + 1e-8)
    rows -= np.outer(rows @ u, u)                             # Qwen3 gate has no bias: rows must ignore shared mean
    rows /= np.linalg.norm(rows, axis=1, keepdims=True) + 1e-8
    logits = np.concatenate(xs) @ rows.T
    return rows * (logit_std / (logits.std() + 1e-8))
