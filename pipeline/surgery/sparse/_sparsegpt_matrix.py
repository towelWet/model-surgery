# _sparsegpt_matrix.py — SparseGPT (Frantar & Alistarh 2023) one-shot pruning of one linear.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Numpy port of IST-DASLab/sparsegpt `SparseGPT.fasterprune` (Apache-2.0): columns processed left
# to right in blocks; mask chosen per block by OBS saliency w² / [H⁻¹]²_jj (adaptive mask
# selection), and each pruned weight's error is pushed onto later columns via upper Cholesky of
# H⁻¹ — same machinery as GPTQ (wquant/_gptq_matrix.py), with "round to grid" replaced by "zero".
# N:M: every M columns pick N lowest-saliency entries per row to prune (reference prunen/prunem).
import numpy as np


def sparsegpt_matrix(w, h, sparsity, nm=None, block=128, damp=0.01):
    """w [out, in], h [in, in] = 2XᵀX/N; nm = (keep_n, m) or None -> pruned float32 w."""
    w = w.astype(np.float64).copy()
    h = h.astype(np.float64).copy()
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0                                          # input never active: weight irrelevant
    h[np.diag_indices_from(h)] += damp * np.mean(np.diag(h))  # 1 % damping keeps Cholesky stable
    hinv = np.linalg.cholesky(np.linalg.inv(h)).T             # upper Cholesky of H⁻¹ (reference order)
    prune_n, m = (nm[1] - nm[0], nm[1]) if nm else (0, 0)    # reference prunes n of every m
    n_in = w.shape[1]
    for i1 in range(0, n_in, block):
        i2 = min(i1 + block, n_in)
        w1, err1, hi = w[:, i1:i2].copy(), np.zeros((w.shape[0], i2 - i1)), hinv[i1:i2, i1:i2]
        if not prune_n:                                       # unstructured: one threshold per block
            sal = w1 ** 2 / np.diag(hi)[None, :] ** 2
            mask = sal <= np.sort(sal.ravel())[int(sal.size * sparsity)]
        else:
            mask = np.zeros_like(w1, dtype=bool)
        for i in range(i2 - i1):
            if prune_n and i % m == 0:                        # N:M group starts: choose its pruned set
                sal = w1[:, i:i + m] ** 2 / np.diag(hi)[i:i + m][None, :] ** 2
                idx = np.argsort(sal, axis=1)[:, :prune_n]
                np.put_along_axis(mask[:, i:i + m], idx, True, axis=1)
            q = np.where(mask[:, i], 0.0, w1[:, i])
            e = (w1[:, i] - q) / hi[i, i]
            w1[:, i:] -= np.outer(e, hi[i, i:])               # OBS update of remaining block columns
            w1[:, i] = q
            err1[:, i] = e
        w[:, i1:i2] = w1
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]                  # lazy batch update of later blocks
    return w.astype(np.float32)
