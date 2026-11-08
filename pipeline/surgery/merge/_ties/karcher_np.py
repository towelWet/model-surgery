# ../merge/_ties/karcher_np.py — Karcher (Riemannian) mean of N tensors on unit sphere, rescaled by mean norm.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Generalises SLERP to N models: iterate log-map average at current mean, exp-map back,
# until step < tol. Magnitude = weighted mean of input norms (direction and size separately).
import numpy as np


def karcher_np(others, weights, iters=10, tol=1e-5):
    w = np.asarray(weights, dtype=np.float64) / float(sum(weights))
    flats = [o.astype(np.float64).ravel() for o in others]
    norms = np.array([np.linalg.norm(f) for f in flats])
    units = [f / n if n > 0 else f for f, n in zip(flats, norms)]
    mu = sum(wi * u for wi, u in zip(w, units))
    mu /= np.linalg.norm(mu)
    for _ in range(iters):
        step = np.zeros_like(mu)
        for wi, u in zip(w, units):
            cos = np.clip(mu @ u, -1.0, 1.0)
            theta = np.arccos(cos)
            if theta > 1e-9:
                step += wi * theta / np.sin(theta) * (u - cos * mu)  # log map of u at mu
        size = np.linalg.norm(step)
        if size < tol:
            break
        mu = np.cos(size) * mu + np.sin(size) * step / size    # exp map
    return (mu * float(w @ norms)).reshape(others[0].shape).astype(np.float32)
