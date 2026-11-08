# _cache_first.py — run cache_teacher_logits.py (optionally swapping teacher in and back out).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import subprocess
import sys


def cache_first(args, cache):
    script = os.path.join(os.path.dirname(__file__), "..", "cache_teacher_logits.py")
    cmd = [sys.executable, script, "--data", args.data, "--out", cache]
    if args.teacher:
        cmd += ["--teacher", args.teacher, "--serve"]       # swap teacher onto tensor-array first
    if args.restore:
        cmd += ["--restore", args.restore]                  # old code always passed it: None in argv crashed
    print("Caching teacher logits:", " ".join(cmd))
    subprocess.run(cmd, check=True)
