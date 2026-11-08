# position/weight.py — per-position distill loss weight: final token 2.0, last 8 tokens 1.5, body 1.0.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def position_weight(pos, n_pos):
    if pos == n_pos - 1:
        return 2.0                                         # EOS / closing token: stop behaviour matters most
    if pos >= max(0, n_pos - 8):
        return 1.5                                         # tail ramp: tool-call closing syntax
    return 1.0
