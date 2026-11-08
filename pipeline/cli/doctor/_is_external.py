# _is_external.py — True when a top-level module comes from stdlib / site-packages, not this repo.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import importlib.util
import sys

_cache = {}


def is_external(top):
    if top not in _cache:
        if top in sys.stdlib_module_names:
            _cache[top] = True
        else:
            try:
                _cache[top] = importlib.util.find_spec(top) is not None  # top-level: no code executed
            except (ImportError, ValueError):
                _cache[top] = False
    return _cache[top]
