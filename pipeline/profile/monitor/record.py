import json, os, sys
from datetime import datetime
try:
    j = json.loads(os.environ.get("RESP") or "{}")
except Exception:
    j = {}
text = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
n = int((j.get("usage") or {}).get("completion_tokens") or 0)
dt = max(float(os.environ["T1"]) - float(os.environ["T0"]), 1e-9)
tps = n / dt
warn = tps < float(os.environ["THRESH"])
rec = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"),
       "tok_s": round(tps, 3), "snippet": text[:20], "warn": warn}
open(os.environ["LOG"], "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
if warn:
    print("WARNING: tok/s %.3f below %s" % (tps, os.environ["THRESH"]), file=sys.stderr)
