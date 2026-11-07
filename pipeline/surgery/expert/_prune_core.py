# Stream pruned expert groups instead of retaining a full output model in unified memory.
from checkpoint.write_groups import write_groups
from shard_io import load_index
from pruned_groups import pruned_groups


def prune_experts(src, dest, plan, cfg, n_experts):
    index = load_index(src)
    groups = pruned_groups(src, index["weight_map"], plan, n_experts)
    write_groups(src, dest, groups, index, cfg)
