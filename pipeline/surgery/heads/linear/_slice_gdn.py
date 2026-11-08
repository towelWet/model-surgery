# ../heads/linear/_slice_gdn.py — apply gdn_index to one layer's linear_attn tensors (bf16 only).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# norm.weight is per-dv and shared by all heads, so it is copied unchanged (not listed here).
import mlx.core as mx

# tensor suffix -> (index key from gdn_index, axis): out_proj loses input columns, rest lose rows
PLAN = {"in_proj_qkv.weight": ("qkv", 0), "conv1d.weight": ("qkv", 0),
        "in_proj_z.weight": ("vd", 0), "out_proj.weight": ("vd", 1),
        "in_proj_a.weight": ("v", 0), "in_proj_b.weight": ("v", 0),
        "A_log": ("v", 0), "dt_bias": ("v", 0)}


def slice_gdn(w, prefix, idx):
    """w: key -> mx array; prefix ends in "linear_attn."; idx from gdn_index -> sliced dict."""
    out = {}
    for name, (which, axis) in PLAN.items():
        key = prefix + name
        out[key] = mx.take(w[key], mx.array(idx[which]), axis=axis)
        mx.eval(out[key])                                       # free lazy source rows early
    return out
