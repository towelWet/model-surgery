"""(1+1)-ES finds minimum of cheap quadratic within budget; mutation respects bounds."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np


def test_es_descends_and_bounds():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.evolve._evolve import evolve
    from pipeline.surgery.evolve._mutate import mutate
    target = {"weights": [0.3, 1.4], "density": [0.7], "lambda": [0.9]}
    f = lambda g, i: sum((np.array(g[k]) - np.array(target[k])) @ (np.array(g[k]) - np.array(target[k])) for k in g)
    start = {"weights": [1.0, 1.0], "density": [0.5], "lambda": [1.0]}
    best, score, log = evolve(start, f, 60, seed=1)
    assert score < 0.25 * log[0]["score"] and len(log) == 60  # big improvement, exact budget
    g = mutate({"weights": [1.99, 0.01], "density": [0.06], "lambda": [0.0]}, np.random.default_rng(0), 5.0)
    assert all(0 <= w <= 2 for w in g["weights"]) and 0.05 <= g["density"][0] <= 1.0
