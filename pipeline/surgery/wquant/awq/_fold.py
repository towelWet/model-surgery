# awq/_fold.py — apply AWQ scales to one layer's float weights and fold 1/s into preceding ops.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# q/k/v input = RMSNorm(x)·ln1 -> ln1 /= s; gate/up input -> ln2 /= s; down input = silu(g)·u
# -> up rows /= s. Folds are exact in float, so only quantization error changes.
# o_proj skipped: its input channels belong to q heads sharing GQA kv heads, so a per-channel
# fold into v_proj would need equal scales across each kv group.
from pipeline.surgery.wquant.awq._scale import awq_scale


def awq_fold(t, xs, bits, group):
    s = awq_scale([t["q"], t["k"], t["v"]], xs["q"], bits, group)
    for r in ("q", "k", "v"):
        t[r] = t[r] * s
    t["ln1"] = t["ln1"] / s
    s = awq_scale([t["gate"], t["up"]], xs["gate"], bits, group)
    t["gate"], t["up"], t["ln2"] = t["gate"] * s, t["up"] * s, t["ln2"] / s
    s = awq_scale([t["down"]], xs["down"], bits, group)
    t["down"], t["up"] = t["down"] * s, t["up"] / s[:, None]  # up rows = down input channels
    return t
