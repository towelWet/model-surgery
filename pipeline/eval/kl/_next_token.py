# _next_token.py — sample one token with top-k logprobs from tensor-array.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.profile._api import _post
from pipeline.eval.kl._prompts import TOP_K


def _next_token(messages):
    """Request one completion token with logprobs from tensor-array."""
    body = {
        "model": "current",
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0.0,
        "enable_thinking": False,
        "logprobs": True,
        "top_logprobs": TOP_K,
    }
    resp = _post("chat/completions", body)
    lp = resp["choices"][0]["logprobs"]["content"][0]
    topk = {t["token"]: t["logprob"] for t in lp["top_logprobs"]}
    return lp["token"], topk
