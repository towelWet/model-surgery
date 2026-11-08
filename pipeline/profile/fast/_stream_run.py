# _stream_run.py — one streamed "Count 1 2 3 4 5." request: time-to-first-token + decode tok/s.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Counts SSE content/reasoning deltas as tokens (server streams ~1 token per delta).
import io
import json
import time
import urllib.request


def one_run(port):
    body = json.dumps({
        "model": "current",                                    # whatever tensor-array is serving
        "messages": [{"role": "user", "content": "Count 1 2 3 4 5."}],
        "max_tokens": 32, "temperature": 0.0, "enable_thinking": False,  # greedy, no think block
        "stream": True,
    }).encode()
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t_post = time.perf_counter(); t_first = t_last = None; n = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in io.TextIOWrapper(resp, encoding="utf-8"):
            line = raw.strip()
            if not line.startswith("data:"):                   # skip SSE keepalives / blank lines
                continue
            payload = line[5:].strip(); now = time.perf_counter()
            if payload == "[DONE]":
                t_last = now; break
            d = ((json.loads(payload).get("choices") or [{}])[0].get("delta") or {})
            if d.get("content") or d.get("reasoning_content"):
                n += 1; t_first = now if t_first is None else t_first
    ttft = (t_first or t_post) - t_post                        # includes prefill
    decode = (n - 1) / max((t_last or t_post) - (t_first or t_post), 1e-6) if n > 1 else 0.0  # first token excluded
    return ttft, decode
