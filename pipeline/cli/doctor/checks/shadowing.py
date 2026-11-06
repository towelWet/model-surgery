# shadowing.py — doctor check: local module names that a stdlib/builtin module shadows (or vice versa).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Builtins (_io, _json …) are found before sys.path, so a local _io.py is silently unreachable;
# a local json.py/profile.py would hijack stdlib for every sibling. Both are bugs.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # cli/
from _py_files import py_files

STD = set(sys.stdlib_module_names) | set(sys.builtin_module_names)


def run(root):
    return [f"{os.path.relpath(p, root)}: '{p.stem}' is also a stdlib/builtin module"
            for p in py_files(root)
            if p.stem in STD and not (p.parent / "__init__.py").exists()  # pkg members are pkg.name
            and "cases" not in p.parts]


CHECK = {"name": "shadowing", "help": "no local module named like a stdlib/builtin module", "run": run}
