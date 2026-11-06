import mlx.core as mx


def emit(out, interval, key, tensor):
    out[key] = tensor
    if len(out) % interval == 0:
        mx.eval(*out.values())
