"""Bounded coherence/speed smoke check against an already running local server."""
import argparse
import json
import time
import tomllib
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path)
args = parser.parse_args()
config = tomllib.loads(args.config.read_text())
base = config["base_url"]
deadline = time.monotonic() + config["ready_timeout"]
while True:
    try:
        with urllib.request.urlopen(base + "/models", timeout=3) as response:
            json.load(response)
        break
    except OSError:
        if time.monotonic() >= deadline:
            raise SystemExit("server readiness timeout")
        time.sleep(3)
body = {"model": "current", "messages": [{"role": "user", "content": config["prompt"]}],
        "max_tokens": config["max_tokens"], "temperature": 0, "enable_thinking": False}
request = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
started = time.perf_counter()
with urllib.request.urlopen(request, timeout=config["request_timeout"]) as response:
    result = json.load(response)
wall = time.perf_counter() - started
message = result["choices"][0]["message"]
text = message.get("content", "").strip()
reason = message.get("reasoning") or message.get("reasoning_content") or ""
tokens = result.get("usage", {}).get("completion_tokens", 0)
speed = tokens / max(wall, 1e-9)
pos, ordered = -1, True
for digit in config["expected_order"]:
    pos = text.find(digit, pos + 1)
    ordered = ordered and pos >= 0
print(f"tokens={tokens} wall={wall:.3f}s tok/s={speed:.2f}")
print(f"preview={text[:64]!r}")
ok = speed > config["min_tok_s"] and ordered and text.startswith(config["expected_order"][0])
if config["reject_reasoning"]:
    ok = ok and not reason.strip() and "<think" not in text.lower()
raise SystemExit(0 if ok else 1)
