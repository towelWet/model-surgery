"""Dropping one KV group removes its rows from q/k/v and its query-head columns from o_proj."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx


def test_slice_attention_keeps_right_heads():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.heads._slice_attention import slice_attention
    d, hid = 2, 3                                            # head_dim 2, hidden 3; 4 q heads, 2 kv (rep 2)
    q = np.arange(8 * hid).reshape(8, hid).astype(np.float32)
    w = {"L.self_attn.q_proj.weight": mx.array(q), "L.self_attn.k_proj.weight": mx.array(q[:4]),
         "L.self_attn.v_proj.weight": mx.array(q[:4]), "L.self_attn.o_proj.weight": mx.array(q.T.copy())}
    out = {k: np.array(v) for k, v in slice_attention(w, "L.", [1], 2, d).items()}  # keep group 1 only
    assert np.array_equal(out["L.self_attn.q_proj.weight"], q[4:8])  # q heads 2,3 = rows 4..7
    assert np.array_equal(out["L.self_attn.k_proj.weight"], q[2:4])  # kv head 1 = rows 2..3
    assert np.array_equal(out["L.self_attn.o_proj.weight"], q.T[:, 4:8])
