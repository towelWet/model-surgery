# _quant_pack.py — MLX affine-quantize one tensor into weight/scales/biases keys.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# mlx.core only (array library), never mlx_lm. Same packing the runtime reads for every
# other quantized tensor: uint32 weight, per-group scales and biases along the last axis.
import mlx.core as mx


def quant_pack(prefix, w, group, bits):
    """{prefix.weight, prefix.scales, prefix.biases} for float array w (last axis = input)."""
    q, s, b = mx.quantize(mx.array(w).astype(mx.bfloat16), group_size=group, bits=bits)
    return {f"{prefix}.weight": q, f"{prefix}.scales": s, f"{prefix}.biases": b}
