# size.py — doctor check: 50-line hard cap and one top-level function/class per file.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # cli/
from _py_files import py_files

CAP = 50                                                    # user hard rule: lines per file
EXEMPT_FACTORY = {"mixed_quant_predicate_builder", "make_emit"}  # closure factories: one unit by design


def run(root):
    out = []
    for p in py_files(root):
        src = p.read_text(errors="ignore")
        rel = os.path.relpath(p, root)
        n = src.count("\n")
        if n > CAP:
            out.append(f"{rel}: {n} lines > {CAP}")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue                                        # imports check reports syntax errors
        units = [x.name for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        if len(units) > 1 and not EXEMPT_FACTORY & set(units):
            out.append(f"{rel}: {len(units)} units {units}")
    return out


CHECK = {"name": "size", "help": f"<= {CAP} lines and one function/class per file", "run": run}
