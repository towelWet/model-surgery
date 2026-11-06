"""Validate one shard header table: __metadata__ str->str, contiguous tensor coverage."""
from .tensor import check_tensor


def check_shard(table, data_size, dtype_bytes_map):
    meta = table.get("__metadata__")
    if meta is not None and (not isinstance(meta, dict)
                             or any(type(k) is not str or type(v) is not str for k, v in meta.items())):
        raise ValueError("__metadata__ must be a string->string object")
    spans = sorted(
        (check_tensor(name, entry, dtype_bytes_map), name)
        for name, entry in table.items() if name != "__metadata__")
    cursor = 0
    for (begin, end), name in spans:
        if begin != cursor:
            raise ValueError(f"{name}: gap or overlap at byte {begin} (expected {cursor})")
        cursor = end
    if cursor != data_size:
        raise ValueError(f"trailing uncovered data: {cursor} != {data_size}")
    return spans
