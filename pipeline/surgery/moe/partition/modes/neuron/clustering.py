# ../../../moe/partition/modes/neuron/clustering.py — neuron-level expert partition (free clustering, needs requant).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from pipeline.surgery._balanced import balanced_kmeans  # qualified; root on sys.path


def neuron_mode(mass, prof, E, S_EXP, SHARED, W):
    """Partition neurons freely — best fidelity but forces down_proj re-quantization."""
    shared = np.sort(np.argsort(-mass)[:SHARED])                            # top-SHARED hottest neurons
    rest = np.setdiff1d(np.arange(W), shared)                               # remaining neurons
    lab = balanced_kmeans(prof.T[rest], E, S_EXP)                           # cluster individually
    experts = np.stack([np.sort(rest[lab == e]) for e in range(E)])
    return shared, experts
