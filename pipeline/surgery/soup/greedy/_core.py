# greedy/_core.py — greedy Model Soup: add fine-tunes best-first, keep each only if score improves.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Wortsman et al. 2022. Lower score = better. Result is never worse than best single model
# on scoring set (it starts from that model). merge(models, out_dir) and score(model_dir)
# are passed in, so this logic is testable without checkpoints.
import os


def greedy_soup(cands, work, merge, score):
    """Returns (kept models, best score, best dir, log rows)."""
    ranked = sorted((score(c), c) for c in cands)             # individual scores, best first
    best, soup, best_dir = ranked[0][0], [ranked[0][1]], ranked[0][1]
    log = [{"model": c, "alone": s} for s, c in ranked]
    for k, (_, c) in enumerate(ranked[1:], 1):
        trial = os.path.join(work, f"trial-{k:02d}")
        merge(soup + [c], trial)                              # uniform average of soup + candidate
        s = score(trial)
        log[k]["soup_with"] = s
        if s < best:
            soup, best, best_dir = soup + [c], s, trial
    return soup, best, best_dir, log
