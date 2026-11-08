# _os_aliases.py — names a file binds to the os module (`import os as _os`), for path replay.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import ast
import os


def os_aliases(tree):
    return {a.asname: os for n in ast.walk(tree) if isinstance(n, ast.Import)
            for a in n.names if a.name == "os" and a.asname}
