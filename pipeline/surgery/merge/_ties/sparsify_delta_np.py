# ../merge/_ties/sparsify_delta_np.py — per-method drop/trim step applied to each task vector before combining.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from .dare_random_np import dare_random_np
from .magnitude_trim_np import magnitude_trim_np
from .breadcrumbs_np import breadcrumbs_np
from .della_np import della_np


def sparsify_delta_np(d, method, density, rng, gamma, epsilon):
    if method == "ties":
        return magnitude_trim_np(d, density)                  # keep top-density magnitudes
    if method in ("dare", "dare_ties"):
        return dare_random_np(d, density, rng)                # random drop + rescale
    if method.startswith("breadcrumbs"):
        return breadcrumbs_np(d, density, gamma)
    if method.startswith("della"):
        return della_np(d, density, epsilon, rng)
    return d                                                  # task_arithmetic: untouched
