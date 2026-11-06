import mlx.core as mx


def project_out(weight, direction, alpha):
    weight = weight.astype(mx.float32)
    return weight - alpha * mx.outer(weight @ direction, direction)
