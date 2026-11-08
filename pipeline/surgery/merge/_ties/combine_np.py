# ../merge/_ties/combine_np.py — TIES sign election + disjoint weighted mean over trimmed task vectors.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np


def ties_combine_np(deltas, weights):
    """deltas: list of already-weighted float32 arrays; returns mixed delta (mergekit 'sum' mask)."""
    total = np.zeros_like(deltas[0])                          # elected-sign mass: sum of weighted deltas
    for d in deltas:
        total += d
    majority = np.where(total >= 0, np.float32(1), np.float32(-1))  # zero mass votes positive, like mergekit
    total.fill(0)                                             # reuse buffer as mixed-delta accumulator
    divisor = np.zeros_like(total)                            # sum of weights that agree per element
    for d, w in zip(deltas, weights):
        agree = np.sign(d) == majority                        # zeroed (trimmed) entries never agree
        total += np.where(agree, d, 0)
        divisor += agree * np.float32(w)
    divisor[divisor == 0] = 1                                 # no agreeing model -> mixed stays 0
    total /= divisor                                          # normalize: disjoint mean, not sum
    return total
