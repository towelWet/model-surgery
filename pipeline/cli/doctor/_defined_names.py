# _defined_names.py — names a module binds at top level (defs, classes, assigns, imports, star-guess).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import ast


def defined_names(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                names |= {e.id for e in ast.walk(t) if isinstance(e, ast.Name)}
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            if any(a.name == "*" for a in n.names):
                names.add("*")                              # star import: can't enumerate, assume ok
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, (ast.If, ast.Try, ast.With)):
            names |= defined_names(ast.Module(body=n.body, type_ignores=[]))  # try/except fallbacks, `with open` loads
    return names
