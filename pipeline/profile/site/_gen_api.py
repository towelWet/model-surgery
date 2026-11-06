# _gen_api.py — long chat completion with usage stats (site bench only).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import time
import urllib.request

DEFAULT_PORT = 8765
DEFAULT_TIMEOUT_S = 900  # full-page gen can exceed 180s tool-eval cap


def complete(messages, max_tokens=4096, session_id=None, timeout_s=DEFAULT_TIMEOUT_S, port=DEFAULT_PORT):
    api = f"http://localhost:{port}/v1/chat/completions"
    body = {
        "model": "current",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "enable_thinking": False,
    }
    if session_id:
        body["session_id"] = session_id
    t0 = time.perf_counter()
    req = urllib.request.Request(
        api,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout_s).read())
    elapsed = time.perf_counter() - t0
    usage = resp.get("usage") or {}
    comp = int(usage.get("completion_tokens") or 0)
    if not comp:
        comp = max(1, len((resp["choices"][0]["message"].get("content") or "").split()))
    return {
        "content": resp["choices"][0]["message"].get("content", ""),
        "usage": usage,
        "elapsed_s": elapsed,
        "completion_tokens": comp,
        "tok_per_s": comp / elapsed if elapsed > 0 else 0.0,
    }
