# _st_rows.py — read n leading-axis rows of one 2-D safetensors tensor by seek (no full load).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np

NP = {"U32": np.uint32, "F16": np.float16, "F32": np.float32}   # dtypes numpy decodes natively


def rows(loc, r0, n):
    """Rows [r0, r0+n) of tensor at loc=(shard, entry, data_start); BF16 decoded by hand."""
    path, e, start = loc
    if len(e["shape"]) != 2 or r0 < 0 or n < 1 or r0 + n > e["shape"][0]:
        raise ValueError("sample must be within a rank-2 tensor")
    if e["dtype"] not in NP and e["dtype"] != "BF16":
        raise ValueError(f"unsupported sample dtype: {e['dtype']}")
    per = int(np.prod(e["shape"][1:])); width = 2 if e["dtype"] in ("F16", "BF16") else 4
    with open(path, "rb") as f:
        f.seek(start + e["data_offsets"][0] + r0 * per * width); raw = f.read(n * per * width)
    if len(raw) != n * per * width:
        raise ValueError("truncated tensor rows")
    if e["dtype"] == "BF16":                                 # sign | 8-bit exp | 7-bit mantissa
        bits = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
        return bits.view("<f4").reshape(n, per)  # preserves subnormals, signed zero, infinities and NaNs
    return np.frombuffer(raw, NP[e["dtype"]]).reshape(n, per)
