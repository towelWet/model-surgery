# bench/_chat.py — single-shot chat call with timing.
import json
import time
import urllib.request


def chat(port, messages, max_tokens=80):
    body = json.dumps(
        {
            "model": "current",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "enable_thinking": False,
        }
    ).encode()
    url = f"http://localhost:{port}/v1/chat/completions"
    t0 = time.perf_counter()
    resp = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}),
            timeout=600,
        ).read()
    )
    elapsed = time.perf_counter() - t0
    u = resp.get("usage") or {}
    comp = int(u.get("completion_tokens") or max_tokens)
    return {"elapsed_s": elapsed, "completion_tokens": comp, "tok_per_s": comp / max(elapsed, 1e-6)}
