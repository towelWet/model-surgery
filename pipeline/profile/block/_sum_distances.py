# _sum_distances.py — summed PruneMe angular distance per block cut, computed by the runtime.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# POST /v1/internals/layer_distance (runtime probe route, env-gated on the server). Hidden
# states stay server-side; only one summed float per cut comes back, in request order.
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile._api import _post


def sum_distances(prompts, starts, max_tokens):
    cuts = [[s, b] for b, ss in starts.items() for s in ss]           # flat [start, size] list
    reply = _post("internals/layer_distance", {"prompts": prompts, "cuts": cuts, "max_tokens": max_tokens})
    if len(reply["totals"]) != len(cuts) or any(v is None for v in reply["totals"]):
        raise ValueError("runtime returned missing or non-finite distances")
    totals = {b: {} for b in starts}
    for (s, b), v in zip(cuts, reply["totals"]):
        totals[b][s] = v
    return totals, reply["prompts"]                                   # server count: short prompts skipped
