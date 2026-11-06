"""Validate one tensor header entry: dtype/shape/offsets vs declared byte width."""


def check_tensor(name, entry, dtype_bytes_map):
    if not isinstance(entry, dict):
        raise ValueError(f"{name}: tensor entry is not an object")
    dtype = entry.get("dtype")
    if not isinstance(dtype, str) or dtype not in dtype_bytes_map:
        raise ValueError(f"{name}: unknown or unsupported dtype {dtype!r}")
    shape = entry.get("shape")
    if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError(f"{name}: invalid shape {shape!r}")
    offsets = entry.get("data_offsets")
    if not (isinstance(offsets, list) and len(offsets) == 2
            and all(type(o) is int and o >= 0 for o in offsets)):
        raise ValueError(f"{name}: invalid data_offsets {offsets!r}")
    begin, end = offsets
    count = 1
    for dim in shape:
        count *= dim
    if end - begin != count * dtype_bytes_map[dtype]:
        raise ValueError(f"{name}: offsets {offsets} do not match {dtype}{shape} bytes")
    return begin, end
