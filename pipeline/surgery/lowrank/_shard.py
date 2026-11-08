# _shard.py — factor one shard's target linears; dense reconstruction or factors in place.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# factored=False: "<key>" = dense A@B (scores anywhere), factors returned for side file.
# factored=True: "<prefix>.lr_a"/".lr_b" replace "<prefix>.weight" in shard itself; runtime
# lowrank_linear serves them, so checkpoint and matmul cost shrink by `ratio`.
from pipeline.surgery.sparse._target import sparse_target
from pipeline.surgery.sparse.layer._norms import layer_norms
from pipeline.surgery.lowrank._factor_one import factor_one


def lowrank_shard(keys, lazy, ctx):
    """ctx: ratio, roles, files, alpha, factored, norms (layer cache, mutated). -> (out, side, rows)."""
    out, side, rows = {}, {}, []
    for key in keys:
        tgt = sparse_target(key)
        if not tgt or tgt[1] not in ctx["roles"]:
            out[key] = lazy[key]; continue
        if ctx["files"] and tgt[0] not in ctx["norms"]:
            ctx["norms"] = {tgt[0]: layer_norms(lazy, key, tgt[0], ctx["files"])}  # one layer cached
        scale = ctx["norms"][tgt[0]][tgt[1]] ** ctx["alpha"] if ctx["files"] else None
        dense, a, b, row = factor_one(key, lazy[key], ctx["ratio"], scale)
        p = key[: -len(".weight")]
        if ctx["factored"]:
            out[p + ".lr_a"], out[p + ".lr_b"] = a, b
        else:
            out[key] = dense
            side[key + ".lr_a"], side[key + ".lr_b"] = a, b
        rows.append(row)
    return out, side, rows
