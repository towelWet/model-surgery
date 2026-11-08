# _channel_importance.py — residual-channel importance: Minitron-style activation magnitude.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Muralidharan et al. 2024 rank embedding channels by summed |normed activation| over all
# layers. Captured "attn.<i>" and "mlp.<i>" are exactly those normed activations.
import numpy as np
from pipeline.surgery.moefy._load_calib import load_calib


def channel_importance(files, n_layers):
    """[H] mean |activation| per residual channel, summed over both norms of every layer."""
    total = None
    for li in range(n_layers):
        for tap in ("attn", "mlp"):
            v = np.abs(load_calib(files, li, tap)).mean(0)
            total = v if total is None else total + v
    return total
