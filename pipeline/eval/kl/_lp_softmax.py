# _lp_softmax.py — renormalize a {token: logprob} top-k dict into probabilities summing to 1.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import math


def lp_softmax(lp):
    m = max(lp.values())                                     # subtract max: overflow-safe exp
    z = sum(math.exp(v - m) for v in lp.values())
    return {k: math.exp(lp[k] - m) / z for k in lp}
