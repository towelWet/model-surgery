# _gate_up.py — slice gate_proj/up_proj into shared + expert tensors.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def partition_gate_up(tensor, E, S, H):
    """Slice gate_proj or up_proj [W, ...] into shared + E expert tensors.

    rows 0:H      → shared expert neurons (always-active)
    rows H:H+E*S  → E expert slices of S rows each
    """
    shared = tensor[:H]                                  # always-active shared FFN neurons
    experts = [
        tensor[H + e * S : H + (e + 1) * S]
        for e in range(E)
    ]
    return shared, experts
