# _path_helpers.py — top-level functions in a module whose body edits sys.path (callable helpers).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import ast

from pipeline.cli.doctor._is_sys_path_call import is_sys_path_call


def path_helpers(tree, path):
    return {n.name: (n, str(path)) for n in tree.body
            if isinstance(n, ast.FunctionDef) and any(is_sys_path_call(x) for x in ast.walk(n))}
