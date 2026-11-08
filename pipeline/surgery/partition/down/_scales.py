# down/_scales.py — slice down_proj scales/biases per expert.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def partition_down_scales(tensor, E, S, H, group_size):
    """Slice down_proj scales or biases [hidden, n_groups] per expert.

    Groups are contiguous in the intermediate dimension.
    """
    hg = H // group_size                                 # group cols for shared neurons
    sg = S // group_size                                 # group cols per expert shard
    shared = tensor[:, :hg]
    experts = [
        tensor[:, hg + e * sg : hg + (e + 1) * sg]
        for e in range(E)
    ]
    return shared, experts
