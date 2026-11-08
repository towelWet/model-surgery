# _evolve.py — (1+1) evolution strategy with 1/5 success rule over merge genomes, fixed budget.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Every evaluation is a full merge + held-out score (minutes each), so population methods
# (CMA-ES needs ~4+3ln(n) per generation) do not fit small budgets; (1+1)-ES spends each
# evaluation on one child of current best. Success widens sigma x1.5, failure shrinks x0.82
# (Rechenberg 1/5 rule: equilibrium at ~20 % success). Lower score = better.
import numpy as np
from pipeline.surgery.evolve._mutate import mutate


def evolve(start, evaluate, evals, seed=0, sigma=0.3):
    rng = np.random.default_rng(seed)
    best, best_score = start, evaluate(start, 0)
    log = [{"eval": 0, "genome": start, "score": best_score, "kept": True}]
    for i in range(1, evals):
        child = mutate(best, rng, sigma)
        s = evaluate(child, i)
        kept = s < best_score
        log.append({"eval": i, "genome": child, "score": s, "kept": kept, "sigma": sigma})
        if kept:
            best, best_score = child, s
        sigma *= 1.5 if kept else 0.82
    return best, best_score, log
