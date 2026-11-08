# expert/_norms.py — per-token output size of each expert's slice of a dense MLP.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np


def expert_norms(acts, wd, labels, experts):
    """[N, E]: L2 norm of acts[:, expert e's neurons] @ wd[:, those neurons].T — router target."""
    cols = [np.linalg.norm(acts[:, labels == e] @ wd[:, labels == e].T, axis=1) for e in range(experts)]
    return np.stack(cols, axis=1)
