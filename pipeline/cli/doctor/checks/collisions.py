# collisions.py — doctor check: bare-import module names defined in 2+ dirs (sys.path order picks one).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Package members (dir has __init__.py) are imported as pkg.mod, so they can't collide; skip them.
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # cli/
from _py_files import py_files



def run(root):
    by = collections.defaultdict(list)
    for p in py_files(root):
        if (p.parent / "__init__.py").exists() or "cases" in p.parts or "cli" in p.parts:
            continue                                        # cases/cli load by file path, not by name
        if not p.stem.startswith("_"):
            continue                                        # no underscore = entry script, run not imported
        by[p.stem].append(os.path.relpath(p, root))
    return [f"{stem}: {', '.join(paths)}" for stem, paths in sorted(by.items())
            if len(paths) > 1]


CHECK = {"name": "collisions", "help": "module basenames unique across bare-import dirs", "run": run}
