# ../merge/_ties/multi_np.py — base + N fine-tunes: DARE sum, TIES, or DARE-TIES (host float32 numpy).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np

from .combine_np import ties_combine_np
from .sparsify_delta_np import sparsify_delta_np
from .whole_model_np import WHOLE, whole_model_np

# *_ties / della: sign election + disjoint mean; dare, breadcrumbs, della_linear, task_arithmetic: sum.
# WHOLE (linear soup, slerp, model_stock, nuslerp, karcher) combine full weights: whole_model_np.
METHODS = ("dare", "ties", "dare_ties", "task_arithmetic", "breadcrumbs", "breadcrumbs_ties",
           "della", "della_linear") + WHOLE
_SUM = ("dare", "task_arithmetic", "breadcrumbs", "della_linear")


def merge_multi_np(base_w, others, method, density, weights, lam, rng, gamma=0.01, epsilon=0.1):
    """lam doubles as SLERP t (0 = base, 1 = fine-tune)."""
    if method not in METHODS:
        raise ValueError(f"merge method must be one of {METHODS}")
    if len(others) != len(weights) or not others:
        raise ValueError("need one weight per fine-tune and at least one fine-tune")
    if not all(np.isfinite(v) for v in (*weights, lam)):
        raise ValueError("merge weights must be finite")
    if any(o.shape != base_w.shape for o in others):
        raise ValueError("merge tensor shapes must match exactly")
    if method in WHOLE:
        return whole_model_np(base_w, others, method, weights, lam)
    deltas = []
    for other, w in zip(others, weights):
        d = np.subtract(other, base_w, dtype=np.float32)    # task vector
        d = sparsify_delta_np(d, method, density, rng, gamma, epsilon)
        d *= np.float32(w)
        deltas.append(d)
    if method in _SUM:
        mixed = deltas[0]
        for d in deltas[1:]:
            mixed += d                                        # task arithmetic: weighted sum
    else:
        mixed = ties_combine_np(deltas, weights)            # sign election + disjoint mean
    mixed *= np.float32(lam)
    mixed += base_w
    return mixed
