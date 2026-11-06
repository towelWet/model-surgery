# _corpus.py — offline, non-personal calibration text for MLP activation profiling.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Three sources, chosen to resemble real god traffic without touching any user data:
#   tool chats  — CALIB prompts rendered WITH the tool schema through the chat template
#   code        — Python standard-library source (ships with every interpreter)
#   prose       — macOS man pages rendered to plain text
# Returns fixed-length token chunks so every forward pass has identical shape.
from calib._tool_chats import _tool_chats
from calib._stdlib_code import _code
from calib._man_prose import _prose


def chunks(tok, length=256, per_source=16):
    """Token chunks of exactly `length`, at most `per_source` from each source."""
    result = []
    for texts in (_tool_chats(tok), _code(), _prose()):
        ids = [i for t in texts for i in tok.encode(t)]            # concatenate within a source
        n = min(per_source, len(ids) // length)
        result += [ids[k * length:(k + 1) * length] for k in range(n)]
    return result
