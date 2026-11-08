# size.py — 50-line hard cap and one function per file, including methods and closures.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.cli._py_files import py_files

CAP = 50                                                    # user hard rule: lines per file


def run(root):
    out = []
    files = py_files(root) + sorted((Path(root).parent / "tests").rglob("*.py"))
    for p in files:
        src = p.read_text(errors="ignore")
        rel = os.path.relpath(p, root)
        n = len(src.splitlines())
        if n > CAP:
            out.append(f"{rel}: {n} lines > {CAP}")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue                                        # imports check reports syntax errors
        units = [x.name for x in ast.walk(tree) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(units) > 1:
            out.append(f"{rel}: {len(units)} units {units}")
    return out


CHECK = {"name": "size", "help": f"<= {CAP} lines and one function/class per file", "run": run}
