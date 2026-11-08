# ../merge/_ties/whole_model_np.py — merges that combine whole weights (no per-delta drop/elect step).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from .linear_np import linear_np
from .slerp_np import slerp_np
from .model_stock_np import model_stock_np
from .nuslerp_np import nuslerp_np
from .karcher_np import karcher_np

WHOLE = ("linear", "slerp", "model_stock", "nuslerp", "karcher")


def whole_model_np(base_w, others, method, weights, lam):
    """Merged tensor for WHOLE methods; lam is SLERP t (0 = base, 1 = fine-tune)."""
    if method == "linear":
        return linear_np(others, weights)
    if method == "slerp":
        return slerp_np(base_w, others[0], lam)
    if method == "model_stock":
        return model_stock_np(base_w, others)
    if method == "nuslerp":
        return nuslerp_np(base_w, others[0], others[1], weights)
    return karcher_np(others, weights)
