# ../merge/_ties/breadcrumbs_np.py — Model Breadcrumbs mask: drop largest outliers AND smallest deltas.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Davari & Belilovsky 2023: top `gamma` fraction by magnitude are outliers that disturb the
# merge; bottom ones are noise. Keeps `density` fraction between them; no rescale. Mutates delta.
import numpy as np


def breadcrumbs_np(delta, density, gamma):
    if not 0 < density <= 1 or not 0 <= gamma < 1 or density + gamma > 1:
        raise ValueError("breadcrumbs needs 0 < density, 0 <= gamma, density + gamma <= 1")
    flat = delta.reshape(-1)
    order = np.argsort(np.abs(flat))                          # ascending magnitude
    n = flat.size
    top, keep = int(gamma * n), int(density * n)
    flat[order[n - top:]] = 0                                 # outliers
    flat[order[: n - top - keep]] = 0                         # small deltas
    return delta
