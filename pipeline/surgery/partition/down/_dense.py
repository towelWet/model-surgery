# _partition_down_dense.py — unquantized down_proj column split (Empero bf16).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def partition_down_dense(tensor, E, S, H):
    """Slice dense down_proj [hidden, W] into shared + E expert column blocks."""
    shared = tensor[:, :H]                   # always-active shared FFN columns
    experts = [
        tensor[:, H + e * S : H + (e + 1) * S]
        for e in range(E)
    ]
    return shared, experts
