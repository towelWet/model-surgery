# _pieces.py — cut offline calibration text into equal-size pieces for MLP-input capture.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Code and prose alternate so every slice of the list (train head, held-out tail) mixes both.
from pipeline.profile.calib._stdlib_code import _code                          # Python stdlib source text
from pipeline.profile.calib._man_prose import _prose                           # macOS man pages as plain text


def pieces(count, chars=1200):
    """`count` text pieces of `chars` characters, alternating code / prose, deterministic order."""
    pools = []
    for texts in (_code(), _prose()):
        joined = "\n\n".join(texts)                           # one long stream per source
        pools.append([joined[i:i + chars] for i in range(0, len(joined) - chars, chars)])
    out = []
    for k in range(count):
        pool = pools[k % 2]                                   # even = code, odd = prose
        idx = (k // 2) * 7 % len(pool)                        # stride 7: spread over many files, not one
        out.append(pool[idx])
    return out
