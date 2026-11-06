# Balanced heal — Count/Generate probe

Do **not** hit :8765/:8766 tonight. Post-heal / morning only.

Overnight measured (Tau-56L-balanced-3bit :8765, thinking off, max_tokens=8):
8 tok / 3.28s = **2.4 tok/s**. `eval_every=8`. Reasoning empty.

64-tok Count: same JSON body; **port only** differs.
Tau production `:8765`. Morning Bonsai `:8766` (`TARR_USE_METAL=1` is **server** env, not curl).

```bash
# Tau :8765
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"current","messages":[{"role":"user","content":"Count 1 2 3 4 5."}],"max_tokens":64,"temperature":0,"enable_thinking":false}'

# Bonsai :8766 (identical body; matches db/MORNING_20260921.md)
curl -s http://127.0.0.1:8766/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"current","messages":[{"role":"user","content":"Count 1 2 3 4 5."}],"max_tokens":64,"temperature":0,"enable_thinking":false}'
```

tok/s = `usage.completion_tokens` / wall seconds.
Bonsai env/serve: BONSAI_GATE.md [missing: not on disk 2026-11-06].
`pipeline/profile/fast/bench.py --port 8765` → `List 1 2 3.` max_tokens=64, thinking off.
