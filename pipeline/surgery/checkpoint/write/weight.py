"""Restore the original dense dtype or requantize all three affine components together."""
import mlx.core as mx


def encode_weight(key, dense, options, dtype):
    if not bool(mx.all(mx.isfinite(dense)).item()):
        raise ValueError(f"{key}: edited weight contains non-finite values")
    if options is None:
        result = {key: dense.astype(dtype)}
    else:
        q, scales, biases = mx.quantize(dense.astype(dtype), **options)
        prefix = key.removesuffix(".weight")
        result = {key: q, prefix + ".scales": scales, prefix + ".biases": biases}
    mx.eval(*result.values())
    return result
