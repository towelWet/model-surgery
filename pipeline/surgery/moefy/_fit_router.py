# _fit_router.py — closed-form ridge regression from MLP input to expert importance.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Runtime router is a bias-free linear layer (mlp.gate), so a linear least-squares fit
# to measured expert output norms is the matching model; ridge keeps it stable.
import numpy as np


def fit_router(x, target, ridge=1e-3):
    """x [N, H], target [N, E] -> R [H, E] minimising |x R - target|^2 + ridge*tr(xTx)/H |R|^2."""
    xtx = x.T @ x
    lam = ridge * np.trace(xtx) / xtx.shape[0]                # scale-free: relative to mean diagonal
    return np.linalg.solve(xtx + lam * np.eye(xtx.shape[0]), x.T @ target)
