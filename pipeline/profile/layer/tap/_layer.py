# tap/_layer.py — run one qwen3_5 DecoderLayer by hand, returning MLP input, activations, output.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Mirrors mlx_lm/models/qwen3_5.py DecoderLayer.__call__ (lines 228-240); capture_mlp checks equality.
import mlx.nn as nn


def tap(layer, h, mask, cache):
    mixer = layer.linear_attn if layer.is_linear else layer.self_attn  # hybrid: linear-attn or full-attn
    h2 = h + mixer(layer.input_layernorm(h), mask, cache)           # residual after mixer
    m_in = layer.post_attention_layernorm(h2)
    act = nn.silu(layer.mlp.gate_proj(m_in)) * layer.mlp.up_proj(m_in)  # SwiGLU neuron activations
    return m_in, act, h2 + layer.mlp.down_proj(act)
