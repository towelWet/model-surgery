# module.py — import one pipeline module by file path, isolated from package context.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import importlib.util                                # file-path import: avoids sys.path pollution
import sys                                         # forbidden-package check reads sys.modules
from pathlib import Path                           # repo-root resolution

ROOT = Path(__file__).resolve().parents[3]         # tests/profile/helpers/module.py -> root
FORBIDDEN = ("mlx_lm", "mlx_vlm")                  # banned imports (severity ∞); also flag bare mlx


def load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                   # must not pull mlx*/mlx_lm at import time
    bad = [m for m in sys.modules if m == "mlx" or m.startswith(FORBIDDEN)]
    assert not bad, f"forbidden modules imported: {bad}"
    return mod
