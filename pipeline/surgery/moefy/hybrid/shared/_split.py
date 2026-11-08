# ../moefy/hybrid/shared/_split.py — pick shared-expert neurons: largest mean output contribution on calibration tokens.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Shared expert runs on every token, so neurons that matter for most tokens go there and
# routing only has to choose among the rest (shared-expert MoEfication, as in Qwen/DeepSeek MoE).
import numpy as np


def shared_split(acts, wd, width):
    """acts [N, I], wd [H, I] -> (shared ids, routed ids), both ascending; shared has `width` ids."""
    contrib = np.abs(acts).mean(0) * np.linalg.norm(wd, axis=0)  # mean |a_i| x |down column i|
    shared = np.sort(np.argsort(-contrib)[:width])
    routed = np.setdiff1d(np.arange(acts.shape[1]), shared)      # sorted complement
    return shared, routed
