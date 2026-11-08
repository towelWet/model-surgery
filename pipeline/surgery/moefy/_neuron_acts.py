# _neuron_acts.py — SwiGLU hidden activations of a dense MLP for given inputs.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np


def neuron_acts(x, wg, wu):
    """[N, I] = silu(x @ wg.T) * (x @ wu.T); wg, wu are [I, H] float32 (HF layout)."""
    g = x @ wg.T                                              # gate pre-activation
    sig = 0.5 * (1.0 + np.tanh(0.5 * g))                     # overflow-free sigmoid (exp(-g) hit inf at g << 0)
    return g * sig * (x @ wu.T)                               # silu(g) * up — exactly what down_proj sees
