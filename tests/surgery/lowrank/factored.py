"""lowrank --factored: shard holds lr_a/lr_b instead of weight; A@B equals dense reconstruction."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def test_factored_shard_replaces_weight_with_factors():
    import mlx.core as mx
    from pipeline.surgery.lowrank._shard import lowrank_shard
    key = "model.layers.0.mlp.up_proj.weight"
    w = mx.random.normal((256, 128), key=mx.random.key(2)).astype(mx.bfloat16)
    lazy = {key: w, "model.norm.weight": mx.ones((128,))}
    ctx = {"ratio": 0.5, "roles": {"up"}, "files": [], "alpha": 0.5, "norms": {}}
    dense, side, _ = lowrank_shard(list(lazy), lazy, {**ctx, "factored": False})
    fac, side2, rows = lowrank_shard(list(lazy), lazy, {**ctx, "factored": True})
    p = key[: -len(".weight")]
    assert key not in fac and {p + ".lr_a", p + ".lr_b"} <= set(fac) and not side2
    assert "model.norm.weight" in fac                         # non-target tensors pass through
    rec = (fac[p + ".lr_a"].astype(mx.float32) @ fac[p + ".lr_b"].astype(mx.float32))
    assert mx.abs(rec - dense[key].astype(mx.float32)).max().item() < 0.05  # same factors, bf16 rounding
    r = rows[0]["rank"]
    assert fac[p + ".lr_a"].shape == (256, r) and r * (256 + 128) <= 0.5 * 256 * 128 + 64 * 384  # ~ratio params
