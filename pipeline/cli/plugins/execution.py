# execution.py — execute one plugin source file fresh in a throwaway module.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# compiles from disk text (no pyc reuse) and never registers in sys.modules,
# so every load_plugins() call sees creates/edits/removes since the last call.
import types


def exec_fresh(path, modname):
    """Run path's source in a fresh module and return it; __file__ set for path-relative plugins."""
    mod = types.ModuleType(modname)
    mod.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod
