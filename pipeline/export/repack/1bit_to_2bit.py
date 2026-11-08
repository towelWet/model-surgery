#!/usr/bin/env python3
"""Repack 1-bit MLX weights to 2-bit for native Metal quantized_matmul.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
Was unrunnable: relative import (from ._repack_core) in a script, and that module never existed.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.export.repack._model import repack_model

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Repack 1-bit MLX to 2-bit")
    p.add_argument("src", help="Source 1-bit checkpoint dir")
    p.add_argument("dst", help="Destination 2-bit checkpoint dir")
    a = p.parse_args()
    repack_model(a.src, a.dst)
