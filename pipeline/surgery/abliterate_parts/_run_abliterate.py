# Output-space projection with bounded source/output residency.
from checkpoint.destination import validate_destination
from checkpoint.write_groups import write_groups
from shard_io import load_index
from _data import TARGET_ENDS
from stream import projected_groups


def run_abliterate(src, dest, r_hat, alpha, cfg):
    validate_destination(dest, sources=[src])
    index = load_index(src)
    mapping = index["weight_map"]
    if not any(key.endswith(TARGET_ENDS) for key in mapping):
        raise ValueError("no supported output projection weights found")
    write_groups(src, dest, projected_groups(src, mapping, cfg, r_hat, alpha), index, cfg)
