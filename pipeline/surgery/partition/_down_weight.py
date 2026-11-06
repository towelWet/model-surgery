# _down_weight.py — slice quantized down_proj packed weight per expert.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def partition_down_weight(tensor, E, S, H, bits, group_size):
    """Slice down_proj packed weight [hidden, packed_W] per expert.

    Packed columns: weight packs `bits` bits per value into uint32 words.
    """
    sh = H * bits // 32                                  # packed cols for shared neurons
    sc = S * bits // 32                                  # packed cols per expert shard
    shared = tensor[:, :sh]
    experts = [
        tensor[:, sh + e * sc : sh + (e + 1) * sc]
        for e in range(E)
    ]
    return shared, experts
