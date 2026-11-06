# _resolve_one.py — resolve one import from `src`: (target path | "external" | None, missing names).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import ast
import os

from _defined_names import defined_names
from _find_local import find_local
from _is_external import is_external


def resolve_one(src, module, level, names, dirs):
    if level:                                               # relative: walk up from src's package
        base = os.path.dirname(src)
        for _ in range(level - 1):
            base = os.path.dirname(base)
        target = find_local(module, [base]) if module else os.path.join(base, "__init__.py")
    else:
        target = find_local(module, dirs)
        if target is None:
            return ("external" if is_external(module.split(".")[0]) else None), []
    if target is None or not os.path.isfile(target):
        return target, []                                   # missing, or namespace dir (names unchecked)
    have = defined_names(ast.parse(open(target).read()))
    pkg = os.path.dirname(target) if target.endswith("__init__.py") else None
    missing = [n for n in names if n != "*" and "*" not in have and n not in have
               and not (pkg and find_local(n, [pkg]))]      # `from pkg import submodule` is fine
    return target, missing
