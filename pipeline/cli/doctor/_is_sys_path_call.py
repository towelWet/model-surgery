# _is_sys_path_call.py — True for sys.path.insert(...) / sys.path.append(...) call nodes.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import ast


def is_sys_path_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("insert", "append")
            and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "path"
            and "sys" in ast.unparse(node.func.value.value))     # sys / _sys / _sys2 aliases
