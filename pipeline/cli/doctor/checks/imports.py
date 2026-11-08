# imports.py — doctor check: every entry script's import chain resolves (modules + imported names).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.cli._py_files import py_files
from pipeline.cli.doctor._walk_imports import walk_imports


def run(root):
    entries = [str(p) for p in py_files(root)
               if not p.name.startswith("_") and not {"cli", "cases"} & set(p.parts)  # cases = data loaded by runners
               and not (p.parent / "__init__.py").exists()]           # package members are imported, not run
    seen, problems = set(), []
    for e in entries:
        problems += [p for p in walk_imports(e, root, seen) if p not in problems]
    return problems


CHECK = {"name": "imports", "help": "import chains from every entry script resolve", "run": run}
