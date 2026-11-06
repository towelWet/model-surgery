# topk_at_prefix.py — one greedy next-token request to tensor-array, returning top-K logprobs.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "profile"))  # profile/_api
from _api import _post

from ._constants import TOP_K


def topk_at_prefix(messages_prefix, tools=None):
    """Greedy next-token top-K from tensor-array; returns (token_str, {token: logprob})."""
    body = {
        "model": "current",                                # whatever teacher is being served
        "messages": messages_prefix,
        "max_tokens": 1,                                   # one step: we only want the distribution
        "temperature": 0.0,
        "enable_thinking": False,
        "logprobs": True,
        "top_logprobs": TOP_K,
    }
    if tools:
        body["tools"] = tools
    resp = _post("chat/completions", body)
    lp = resp["choices"][0]["logprobs"]["content"][0]
    topk = {t["token"]: t["logprob"] for t in lp["top_logprobs"]}
    return lp["token"], topk
