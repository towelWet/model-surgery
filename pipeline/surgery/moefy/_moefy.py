# Dense-to-MoE conversion is blocked until routing and loader compatibility are proven.
# Partition primitives remain available for offline structural research.
from checkpoint.destination import validate_destination
from _gate import emit_gates


def moefy(src, dest, experts, topk):
    """Refuse unvalidated conversion before writing any checkpoint files."""
    validate_destination(dest, sources=[src])
    emit_gates(None, None, None, experts, None)
