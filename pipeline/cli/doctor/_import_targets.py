# _import_targets.py — (lineno, module, level, names) for every import in a file, incl. lazy ones.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import ast


def import_targets(tree):
    out = []
    for n in ast.walk(tree):                                # walk: function-level imports count too
        if isinstance(n, ast.Import):
            out += [(n.lineno, a.name, 0, []) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out.append((n.lineno, n.module or "", n.level, [a.name for a in n.names]))
    return out
