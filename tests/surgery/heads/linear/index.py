"""GDN head index: kept key groups keep their q/k rows, own value heads, matching z/a/b/out columns."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
import mlx.core as mx


def test_gdn_index_and_slice_follow_key_groups():
    from pipeline.surgery.heads.linear._gdn_index import gdn_index
    from pipeline.surgery.heads.linear._slice_gdn import slice_gdn
    hk, hv, dk, dv = 2, 4, 1, 1                                 # r = 2 value heads per key head
    idx = gdn_index([1], hk, hv, dk, dv)                        # keep key group 1 only
    assert idx["qkv"].tolist() == [1, 3, 6, 7]                  # q1, k1 (2+1), v heads 2,3 (4+2, 4+3)
    assert idx["vd"].tolist() == [2, 3] and idx["v"].tolist() == [2, 3]
    cd = 2 * hk * dk + hv * dv                                  # 8 channels
    p = "L.linear_attn."
    w = {p + "in_proj_qkv.weight": mx.arange(cd * 3).reshape(cd, 3), p + "conv1d.weight": mx.arange(cd).reshape(cd, 1, 1),
         p + "in_proj_z.weight": mx.arange(hv * 3).reshape(hv, 3), p + "out_proj.weight": mx.arange(3 * hv).reshape(3, hv),
         p + "in_proj_a.weight": mx.arange(hv * 3).reshape(hv, 3), p + "in_proj_b.weight": mx.arange(hv * 3).reshape(hv, 3),
         p + "A_log": mx.arange(hv), p + "dt_bias": mx.arange(hv)}
    out = slice_gdn(w, p, idx)
    assert out[p + "conv1d.weight"].reshape(-1).tolist() == [1, 3, 6, 7]
    assert out[p + "out_proj.weight"].shape == (3, 2) and out[p + "out_proj.weight"][0].tolist() == [2, 3]
    assert out[p + "A_log"].tolist() == [2, 3] and out[p + "in_proj_qkv.weight"].shape == (4, 3)
