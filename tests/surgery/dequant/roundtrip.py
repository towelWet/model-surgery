"""Dequant: affine triple -> bf16 equals mx.dequantize; per-path 8-bit override honoured."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx


def test_dequant_tensors_matches_mlx_and_override():
    from pipeline.surgery.dequant._tensors import dequant_tensors
    w = mx.random.normal((64, 128), key=mx.random.key(0))
    q4, s4, b4 = mx.quantize(w, group_size=64, bits=4)
    q8, s8, b8 = mx.quantize(w, group_size=64, bits=8)
    t = {"a.weight": q4, "a.scales": s4, "a.biases": b4, "h.weight": q8, "h.scales": s8,
         "h.biases": b8, "n.weight": mx.ones((128,))}
    cfg = {"quantization": {"group_size": 64, "bits": 4, "h": {"group_size": 64, "bits": 8}}}
    out = dequant_tensors(t, cfg)
    assert sorted(out) == ["a.weight", "h.weight", "n.weight"]    # scales/biases consumed
    ref4 = mx.dequantize(q4, s4, b4, group_size=64, bits=4).astype(mx.bfloat16)
    assert mx.array_equal(out["a.weight"], ref4).item()
    assert mx.abs(out["h.weight"].astype(mx.float32) - w).max().item() < 0.05  # 8-bit read as 8-bit
    assert out["n.weight"].dtype == mx.float32                     # unquantized passthrough
