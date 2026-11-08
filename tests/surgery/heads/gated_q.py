"""Gated q_proj (Qwen3.5): kept head keeps its query AND gate rows; o_proj keeps D columns per head."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path


def test_slice_attention_keeps_gate_rows_with_head():
    import mlx.core as mx
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.heads._slice_attention import slice_attention
    n_q, n_kv, d, hid = 4, 2, 2, 3                              # rep 2 query heads per KV group
    p = "L."
    w = {p + "self_attn.q_proj.weight": mx.arange(n_q * 2 * d).reshape(-1, 1) * mx.ones((1, hid)),
         p + "self_attn.k_proj.weight": mx.arange(n_kv * d).reshape(-1, 1) * mx.ones((1, hid)),
         p + "self_attn.v_proj.weight": mx.arange(n_kv * d).reshape(-1, 1) * mx.ones((1, hid)),
         p + "self_attn.o_proj.weight": mx.ones((hid, 1)) * mx.arange(n_q * d).reshape(1, -1)}
    out = slice_attention(w, p, [1], n_q // n_kv, d, q_rows=2 * d)  # keep group 1 = query heads 2, 3
    assert out[p + "self_attn.q_proj.weight"][:, 0].tolist() == list(range(8, 16))  # heads 2,3 x [q|gate]
    assert out[p + "self_attn.k_proj.weight"][:, 0].tolist() == [2, 3]
    assert out[p + "self_attn.o_proj.weight"][0].tolist() == [4, 5, 6, 7]           # D columns per head
