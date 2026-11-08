# _layer.py — decoder layer index of an MLP weight key, or None if key is not MLP.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Replaces is_mlp_key(), which only knew "model.layers.N": on nested VLM checkpoints
# (Tau "language_model.model.layers.", Empero "model.language_model.layers.")
# MLP-only prune matched nothing, zeroed nothing, and still wrote a "pruned" model.
import re

# MLX affine quant dequantizes as scales*q + biases: zeroing packed .weight alone leaves
# W = biases (not 0). Zero all three so float AND quantized MLPs really become 0.
MLP_SUFFIXES = tuple(f".mlp.{p}.{t}" for p in ("gate_proj", "up_proj", "down_proj")
                     for t in ("weight", "scales", "biases"))


def mlp_layer(key, prefix):
    if not key.endswith(MLP_SUFFIXES):
        return None                                  # attention, norms, scales: untouched
    if key.startswith(prefix):
        head = key[len(prefix):].split(".", 1)[0]
        return int(head) if head.isdigit() else None
    return None                                      # MTP / vision keys are not decoder layers
