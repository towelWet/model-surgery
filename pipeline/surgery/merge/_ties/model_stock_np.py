# ../merge/_ties/model_stock_np.py — Model Stock: interpolate fine-tune average toward base by delta angle.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Jang et al. 2024: with cos = mean pairwise cosine of task vectors and k fine-tunes,
# t = k*cos / (1 + (k-1)*cos); merged = t * mean(fine-tunes) + (1 - t) * base. Per tensor.
import numpy as np


def model_stock_np(base_w, others):
    if len(others) < 2:
        raise ValueError("model stock needs at least two fine-tunes")
    base = base_w.astype(np.float32)
    deltas = [(o.astype(np.float32) - base).ravel() for o in others]
    cos = []
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            den = np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[j])
            cos.append(float(deltas[i] @ deltas[j] / den) if den > 0 else 1.0)  # identical/zero: parallel
    c, k = float(np.mean(cos)), len(others)
    t = k * c / (1 + (k - 1) * c)
    avg = np.mean([o.astype(np.float32) for o in others], axis=0)
    return t * avg + (1 - t) * base
