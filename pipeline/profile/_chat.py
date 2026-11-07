# _chat.py — chat completion via tensor-array API.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _api import _post
from pipeline.runtime.settings import runtime_settings


def chat(messages, tools=None, max_tokens=256, thinking=False, logprobs=False, top_logprobs=0):
    """Chat completion; returns first choice message dict.

    thinking=False disables chain-of-thought (saves ~100-300 tokens).
    Always disable thinking for tool-call eval — reasoning burns token budget
    before tool call is emitted, causing silent truncation.
    """
    body = {
        "model":       runtime_settings()["http"]["model"],
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.0,                              # greedy: eval must be reproducible
    }
    if tools:
        body["tools"] = tools                            # omit key entirely when no tools
    if not thinking:
        # MUST be top-level. tensor-array does NOT forward chat_template_kwargs.
        # Verified 2026-09-16: top-level -> clean tool call in 24 tokens.
        body["enable_thinking"] = False
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs
    resp = _post("chat/completions", body)
    choice = resp["choices"][0]
    msg = choice["message"]
    if logprobs and "logprobs" in choice:
        msg["_logprobs"] = choice["logprobs"]
    return msg
