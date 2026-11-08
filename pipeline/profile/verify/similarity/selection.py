"""Discover supported common matrices; bound default work across the checkpoint."""


def select_tensors(left, right, requested=None, limit=8):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    names = []
    for key in sorted(left.keys() & right.keys()):
        if not key.endswith(".weight"):
            continue
        prefix = key[:-7]
        if not all(prefix + suffix in side for side in (left, right) for suffix in (".scales", ".biases")):
            continue
        if len(left[key][1]["shape"]) == len(right[key][1]["shape"]) == 2:
            names.append(prefix)
    if requested:
        if set(requested) - set(names):
            raise ValueError("requested tensors are absent or not supported quantized matrices")
        return requested
    if not names:
        raise ValueError("no common quantized matrices found")
    if limit and len(names) > limit:
        return [names[i * (len(names) - 1) // max(limit - 1, 1)] for i in range(limit)]
    return names
