# _gate.py — emit zero-init router gate weights for each layer.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # surgery/ parent
from _layer_prefix import decoder_layer_prefix           # Tau vs Empero prefix


def emit_gates(wmap, emit, n_layers, E, H):
    """Emit zero-init gate weights for E experts across all layers."""
    gate_prefix = decoder_layer_prefix(wmap)             # Tau vs Empero nested lang model
    for li in range(n_layers):
        emit(f"{gate_prefix}{li}.mlp.gate.weight", mx.zeros((E, H)))
