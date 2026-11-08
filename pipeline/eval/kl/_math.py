# _math.py — KL on top-k logprob dicts; math only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# HONEST NOTE: earlier comments claimed tail mass goes to an "other" bucket. It does
# not: each side's top-k is renormalized to sum 1, and a token missing on one side
# gets 1e-12. KL here = divergence between renormalized top-k heads, which
# understates true KL when tail mass differs. Kept as-is so scores stay comparable.
import math

from pipeline.eval.kl._lp_softmax import lp_softmax


def kl_topk(parent_lp, student_lp):
    """KL(parent||student) over union of both top-k supports (renormalized heads)."""
    p = lp_softmax(parent_lp)
    s = lp_softmax(student_lp)
    kl = 0.0
    for k in set(parent_lp) | set(student_lp):
        pv = p.get(k, 1e-12)                                 # absent in parent top-k
        sv = max(s.get(k, 1e-12), 1e-12)                     # absent in student top-k
        kl += pv * (math.log(pv) - math.log(sv))
    return kl
