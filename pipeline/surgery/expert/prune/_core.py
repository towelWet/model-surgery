# Stream pruned expert groups instead of retaining a full output model in unified memory.
from pipeline.surgery.checkpoint.write.groups import write_groups
from pipeline.surgery.shard_io import load_index
from pipeline.surgery.expert.pruned_groups import pruned_groups


def prune_experts(src, dest, plan, cfg, n_experts):
    index = load_index(src)
    groups = pruned_groups(src, index["weight_map"], plan, n_experts)
    write_groups(src, dest, groups, index, cfg)
