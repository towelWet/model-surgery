# _channel_norms.py — per-input-channel L2 norms of real linear-layer inputs (Wanda activation term).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# q/k/v read attention input, o reads per-head outputs, gate/up read MLP input (all captured by
# profile/moefy/capture.py);
# down reads SwiGLU activations, recomputed here from DENSE gate/up (simplification:
# original Wanda re-feeds already-pruned outputs layer by layer).
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.moefy._load_calib import load_calib                          # [N, H] captured MLP inputs
from pipeline.surgery.moefy._neuron_acts import neuron_acts                        # silu(x Wg^T) * (x Wu^T)


def channel_norms(files, layer, wg, wu):
    """Role -> input-channel norm vector over calibration tokens, for all 7 decoder linears."""
    x = load_calib(files, layer)
    nx = np.linalg.norm(x, axis=0)
    na = np.linalg.norm(load_calib(files, layer, "attn"), axis=0)
    no = np.linalg.norm(load_calib(files, layer, "o"), axis=0)
    return {"q": na, "k": na, "v": na, "o": no, "gate": nx, "up": nx,
            "down": np.linalg.norm(neuron_acts(x, wg, wu), axis=0)}
