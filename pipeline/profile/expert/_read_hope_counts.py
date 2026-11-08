# _read_hope_counts.py — tensor-array HOPE stats file -> routing hits + co-activation counts.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Producer: tensor-array src/arch/qwen/3_5/moe/hope_flush.cpp (TARR_HOPE_STATS=<path>).
# Only "P <layer> <i> <j> <count> <pair_sum>" rows matter here:
#   i == j  -> tokens that routed to expert i      (== old per-token top-k hit count)
#   i <  j  -> tokens where i and j were both top-k (stored once; mirrored below)
# "E" rows (MAN/MSAN norm sums) are ignored: hits already come from diagonal P rows.
import os

import numpy as np


def read_hope_counts(path, n_layers, n_experts):
    """Return (hits[L,E], coact[L,E,E]) int64; all zeros when file not written yet."""
    hits = np.zeros((n_layers, n_experts), dtype=np.int64)            # per-expert routed tokens
    coact = np.zeros((n_layers, n_experts, n_experts), dtype=np.int64)  # symmetric co-fire counts
    if not os.path.exists(path):
        return hits, coact                                            # fresh server: nothing recorded
    with open(path) as f:
        for line in f:
            part = line.split()
            if not part or part[0] != "P":
                continue                                              # E rows / blanks: not counts
            li, i, j, cnt = int(part[1]), int(part[2]), int(part[3]), int(part[4])
            if i == j:
                hits[li, i] = cnt                                     # diagonal = expert's own hits
            else:
                coact[li, i, j] = coact[li, j, i] = cnt               # old capture counted both orders
    return hits, coact
