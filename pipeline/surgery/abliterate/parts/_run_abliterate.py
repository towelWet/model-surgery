# Output-space projection with bounded source/output residency.
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.checkpoint.write.groups import write_groups
from pipeline.surgery.shard_io import load_index
from pipeline.surgery.abliterate.parts._data import TARGET_ENDS
from pipeline.surgery.abliterate.parts.stream import projected_groups


def run_abliterate(src, dest, r_hat, alpha, cfg):
    validate_destination(dest, sources=[src])
    index = load_index(src)
    mapping = index["weight_map"]
    if not any(key.endswith(TARGET_ENDS) for key in mapping):
        raise ValueError("no supported output projection weights found")
    write_groups(src, dest, projected_groups(src, mapping, cfg, r_hat, alpha), index, cfg)
