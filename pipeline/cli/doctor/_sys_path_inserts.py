# _sys_path_inserts.py — statically replay a file's sys.path edits: direct calls, literal for-loops,
# and calls to known helper functions (funcs: name -> (FunctionDef, its file)) such as add_eval_paths.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import ast
import os

from pipeline.cli.doctor._eval_expr import eval_expr
from pipeline.cli.doctor._is_sys_path_call import is_sys_path_call


def sys_path_inserts(stmts, ns, funcs, out):
    for st in stmts:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            ns[st.targets[0].id] = eval_expr(st.value, ns)  # _ROOT / _HERE / _PIPE style anchors
        if isinstance(st, ast.For) and isinstance(st.target, ast.Name) and isinstance(st.iter, (ast.Tuple, ast.List)):
            for elt in st.iter.elts:                        # for _d in ("code", "rev"): unrolled
                sys_path_inserts(st.body, {**ns, st.target.id: eval_expr(elt, ns)}, funcs, out)
            continue
        for node in ast.walk(st) if not isinstance(st, (ast.FunctionDef, ast.ClassDef)) else []:
            if is_sys_path_call(node):
                p = eval_expr(node.args[-1], ns)            # insert(0, X) / append(X): X is last
                if isinstance(p, str):
                    out.append(os.path.normpath(p))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
                fdef, ffile = funcs[node.func.id]           # helper defined elsewhere: bind params
                args = {a.arg: eval_expr(v, ns) for a, v in zip(fdef.args.args, node.args)}
                sys_path_inserts(fdef.body, {"__file__": ffile, **args}, funcs, out)
    return out
