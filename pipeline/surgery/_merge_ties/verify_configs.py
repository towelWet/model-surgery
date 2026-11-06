# verify_configs.py — refuse merge unless both checkpoints share architecture dims.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys

from ._shared import ARCH_KEYS
from .text_cfg import text_cfg


def verify_configs(src_a, src_b):
    """Refuse merge when architecture dims diverge."""
    a, b = text_cfg(src_a), text_cfg(src_b)
    for k in ARCH_KEYS:
        if a.get(k) is None or a.get(k) != b.get(k):           # missing key also refuses: can't prove match
            sys.exit(f"config mismatch on {k!r}: src_a={a.get(k)!r}, src_b={b.get(k)!r} — STOP")
    print("config OK:", ", ".join(f"{k}={a[k]}" for k in ARCH_KEYS))
