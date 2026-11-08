# module.py — import one pipeline module by file path, isolated from package context.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import importlib.util                                # file-path import: avoids sys.path pollution
import builtins                                    # __import__ spy: which modules the load requests
import sys
from pathlib import Path                           # repo-root resolution

ROOT = Path(__file__).resolve().parents[3]         # tests/profile/helpers/module.py -> root
FORBIDDEN = ("mlx_lm", "mlx_vlm")                  # banned imports (severity ∞); also flag bare mlx


def load(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    # Earlier tests in same runner process may already hold mlx in sys.modules, so scanning
    # sys.modules is order-dependent. Record every import THIS module requests instead (cached
    # ones too); never pop native modules (nanobind aborts on re-registration).
    seen, real = [], builtins.__import__
    builtins.__import__ = lambda n, *a, **k: (seen.append(n), real(n, *a, **k))[1]  # record, then import
    try:
        spec.loader.exec_module(mod)               # must not pull mlx*/mlx_lm at import time
    finally:
        builtins.__import__ = real
    bad = sorted({n for n in seen if n == "mlx" or n.startswith(("mlx.",) + FORBIDDEN)})
    assert not bad, f"forbidden modules imported: {bad}"
    return mod
