# mlx_lm.py — doctor check: banned mlx_lm / mlx_vlm imports or CLI calls (tensor-array only, ever).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # cli/
from _py_files import py_files

BANNED = re.compile(r"^\s*(from|import)\s+mlx_(lm|vlm)\b|python3? -m mlx_(lm|vlm)|mlx_lm\.\w+\(", re.M)


def run(root):
    files = list(py_files(root)) + sorted(pathlib.Path(root).rglob("*.sh"))   # shell wrappers too
    out = []
    for p in files:
        if "cli" in p.relative_to(root).parts:
            continue                                        # this check's own regex text
        for m in BANNED.finditer(p.read_text(errors="ignore")):
            line = p.read_text(errors="ignore")[: m.start()].count("\n") + 1
            out.append(f"{os.path.relpath(p, root)}:{line} {m.group(0).strip()}")
    return out


CHECK = {"name": "mlx_lm", "help": "no mlx_lm / mlx_vlm anywhere (port to tensor-array)", "run": run}
