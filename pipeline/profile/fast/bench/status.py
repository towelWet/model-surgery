#!/usr/bin/env python3
# bench/status.py — print use_metal from tensor-array GET /api/status (benchmark probe pattern).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import sys
import urllib.request


def show(port):
    try:
        url = f"http://127.0.0.1:{port}/api/status"
        with urllib.request.urlopen(url, timeout=5) as r:
            v = json.load(r).get("use_metal", "?")
    except Exception:
        v = "?"                                                # server down: report unknown, never fail
    print(f"use_metal: {v}")


if __name__ == "__main__":
    show(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)      # 8765 = tensor-array default port
