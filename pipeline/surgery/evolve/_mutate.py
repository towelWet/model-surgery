# _mutate.py — Gaussian mutation of merge genome {weights, density, lambda}, clipped to valid ranges.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np

BOUNDS = {"weights": (0.0, 2.0), "density": (0.05, 1.0), "lambda": (0.0, 2.0)}


def mutate(genome, rng, sigma):
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        v = np.asarray(genome[k], dtype=float)
        out[k] = np.clip(v + rng.normal(0.0, sigma, v.shape), lo, hi).tolist()  # lists stay JSON-able
    return out
