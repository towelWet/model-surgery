# _whittle.py — exact neuron partition E*S + H = W (no invented FFN weights).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Tau-56L:  64 × 192 + 5120 = 17408
# Empero 9B: 64 × 128 + 4096 = 12288


def expert_width(intermediate_size, hidden_size, experts):
    """Return per-expert width S such that experts*S + hidden_size == intermediate_size."""
    remainder = intermediate_size - hidden_size
    if remainder <= 0 or remainder % experts != 0:
        raise ValueError(
            f"cannot partition W={intermediate_size} H={hidden_size} E={experts}"
        )
    width = remainder // experts
    if experts * width + hidden_size != intermediate_size:
        raise ValueError(f"width {width} x {experts} + {hidden_size} != {intermediate_size}")
    return width
