# _walk_imports.py — follow imports from one entry script the way Python would; collect breakage.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# sys.path model: entry dir + every sys.path edit replayed from visited files (edits persist).
# Repeats passes until no new dirs appear, because a helper (add_eval_paths) is only known after
# the file defining it was visited.
import ast
import os

from _import_targets import import_targets
from _os_aliases import os_aliases
from _path_helpers import path_helpers
from _resolve_one import resolve_one
from _sys_path_inserts import sys_path_inserts


def walk_imports(entry, root, seen):
    dirs, funcs = [os.path.dirname(entry)], {}
    for _ in range(6):                                      # fixpoint over sys.path growth
        n_dirs, n_funcs, problems, todo, visited = len(dirs), len(funcs), [], [entry], set()
        while todo:
            f = todo.pop()
            if f in visited:
                continue
            visited.add(f)
            try:
                tree = ast.parse(open(f).read())
            except SyntaxError as e:
                problems.append(f"{os.path.relpath(f, root)}: syntax error line {e.lineno}")
                continue
            funcs.update(path_helpers(tree, f))
            dirs += [d for d in sys_path_inserts(tree.body, {"__file__": f, **os_aliases(tree)}, funcs, []) if d not in dirs]
            for line, mod, level, names in import_targets(tree):
                target, missing = resolve_one(f, mod, level, names, dirs)
                where = f"{os.path.relpath(f, root)}:{line}"
                if target is None:
                    problems.append(f"{where} cannot find module {'.' * level}{mod}")
                elif missing:
                    problems.append(f"{where} {'.' * level}{mod} lacks {', '.join(missing)}")
                if target not in (None, "external") and os.path.isfile(target):
                    todo.append(target)
        if len(dirs) == n_dirs and len(funcs) == n_funcs:
            break
    seen |= visited                                         # shared across entries: orphan report
    return problems
