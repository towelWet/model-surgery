"""Strict verify of a stored single-tensor file: header fields + size + payload sha256."""
import json
import struct
from pathlib import Path
from ..hashes.region import region_sha256


def verify_tensor_file(path, entry):
    """Return False if missing; raise on any mismatch vs manifest tensor entry."""
    path = Path(path)
    if not path.exists():
        return False
    size = path.stat().st_size
    if size < 8:
        raise ValueError(f"damaged partial file (no overwrite allowed): {path}")
    with open(path, "rb") as stream:
        hlen = struct.unpack("<Q", stream.read(8))[0]
        if hlen > min(size - 8, 10_000_000):                 # header bounded, no payload read
            raise ValueError(f"tensor header length damaged: {path}")
        table = json.loads(stream.read(hlen))
    name = entry["name"]
    expected = {"dtype": entry["dtype"], "shape": entry["shape"],
                "data_offsets": [0, entry["nbytes"]]}
    if table != {name: expected}:
        raise ValueError(f"tensor header fields damaged: {path}")
    if size != 8 + hlen + entry["nbytes"]:
        raise ValueError(f"tensor file size damaged: {path}")
    if region_sha256(path, 8 + hlen, entry["nbytes"]) != entry["sha256"]:
        raise ValueError(f"tensor payload damaged: {path}")
    return True
