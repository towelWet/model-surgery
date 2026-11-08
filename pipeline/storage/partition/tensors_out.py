"""Write every planned single-tensor safetensors file; strict verify on resume."""
from ..files.member import safe_member
from ..files.tensor_verify import verify_tensor_file
from ..formats.safetensors.encode import write_tensor_file
from ..formats.safetensors.header import raw_header


def write_tensors(source, dest, plan, resuming):
    for t in plan["tensors"]:
        target = safe_member(dest, t["file"])
        if target.exists():
            if resuming and verify_tensor_file(target, t):
                continue
            raise ValueError(f"damaged partial file (no overwrite allowed): {target}")
        src = safe_member(source, t["shard"])
        data_start = src.stat().st_size - raw_header(src)[2]
        digest = write_tensor_file(target, t["name"], t["dtype"], t["shape"],
                                   src, data_start + t["offsets"][0], t["nbytes"])
        if digest != t["sha256"]:
            raise ValueError(f"payload hash mismatch writing {t['name']}")
