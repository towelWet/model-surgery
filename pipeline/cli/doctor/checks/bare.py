# bare.py — doctor check: no bare imports of repo modules and no sys.path dirs other than the repo root.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Replaces the old `collisions` check (unique basenames across bare-import dirs). Since 2026-11-07 every
# repo import is absolute (pipeline.* / tests.*), so same-named private modules in different packages
# (surgery/heads/_run.py, surgery/wquant/_run.py) are fine; what must never come back is the bare
# style that made Python keep ONE module per name — sibling imports and dir hacks on sys.path.
import ast
import os
import pathlib
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.cli._py_files import py_files
from pipeline.cli.doctor._eval_expr import eval_expr
from pipeline.cli.doctor._is_sys_path_call import is_sys_path_call


def run(root):
    repo, out = os.path.realpath(os.path.dirname(root)), []    # root = pipeline/; repo = its parent
    for p in py_files(root):
        if "cases" in p.parts:
            continue                                            # cases = data loaded by file path
        tree, rel = ast.parse(p.read_text()), os.path.relpath(p, root)
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)) and not getattr(n, "level", 0):
                for mod in ([a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]):
                    top = os.path.join(p.parent, *mod.split("."))
                    if os.path.isfile(top + ".py") or os.path.isdir(top):
                        out.append(f"{rel}:{n.lineno} bare import of sibling '{mod}' (use pipeline.*)")
            if is_sys_path_call(n):
                v = eval_expr(n.args[-1], {"__file__": str(p), "Path": pathlib.Path, "str": str})
                v = os.path.realpath(str(v)) if v is not None else None
                if v and v != repo and v.startswith(repo + os.sep):
                    out.append(f"{rel}:{n.lineno} sys.path gets repo dir {os.path.relpath(v, repo)} (root only)")
    return out


CHECK = {"name": "bare", "help": "absolute repo imports only; sys.path gets repo root only", "run": run}
