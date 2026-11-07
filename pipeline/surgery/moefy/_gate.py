# _gate.py — reject conversion without a trained router and loader contract.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # surgery/ parent
from _layer_prefix import decoder_layer_prefix           # Tau vs Empero prefix


def emit_gates(wmap, emit, n_layers, E, H):
    """Zero/random gates cannot establish learned expert selection."""
    raise NotImplementedError("dense-to-MoE conversion requires trained routers and a verified loader")
