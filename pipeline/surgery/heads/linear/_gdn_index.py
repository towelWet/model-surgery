# ../heads/linear/_gdn_index.py — kept key-head groups -> row indices for every Gated DeltaNet tensor.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Layout (Qwen3.5/3.6 runtime linear_gdn_inproj + gated_delta_kernel):
#   in_proj_qkv / conv1d channels = [q: Hk*dk | k: Hk*dk | v: Hv*dv]
#   value head j reads key head j // (Hv/Hk), so key group g owns value heads g*r .. g*r+r-1.
#   in_proj_z rows and out_proj columns = value heads x dv; in_proj_a/b, A_log, dt_bias = value heads.
import numpy as np
from pipeline.surgery.heads._keep_rows import keep_rows


def gdn_index(keep, hk, hv, dk, dv):
    """keep: sorted kept key groups -> {"qkv", "vd", "v"} int index arrays."""
    r = hv // hk                                                # value heads per key head
    vheads = [g * r + j for g in keep for j in range(r)]        # value heads owned by kept groups
    q = keep_rows(keep, dk)                                     # q block starts at 0
    k = keep_rows(keep, dk) + hk * dk                           # k block after all q rows
    v = keep_rows(vheads, dv) + 2 * hk * dk                     # v block after q and k
    return {"qkv": np.concatenate([q, k, v]),                  # also conv1d channel order
            "vd": keep_rows(vheads, dv),                        # z rows / out_proj columns
            "v": np.array(vheads)}                              # per-value-head scalars
