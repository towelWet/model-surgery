#!/usr/bin/env python3
"""View idle tok/s samples. Wrapper to tensor-array profile/perf-log-view.py."""
import os
import sys

SRC = os.environ.get("TENSOR_ARRAY_DIR", "")                 # tensor-array project root
if SRC:
    SRC = os.path.join(SRC, "profile", "perf-log-view.py")
if not SRC or not os.path.isfile(SRC):
    print("Set TENSOR_ARRAY_DIR to tensor-array project root")
    sys.exit(1)
os.execv(sys.executable, [sys.executable, SRC, *sys.argv[1:]])
