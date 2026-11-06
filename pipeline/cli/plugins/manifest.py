# manifest.py — validate a discovered plugin manifest against the generic contract.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Contract (generic — no plugin names known here): manifest is a dict with a
# non-empty string "name" and a callable "run"; "help" and extra keys optional.


def validate(man, path, attr):
    """Return man if valid, None if file has no manifest attr; raise on a malformed manifest."""
    if man is None:
        return None                                                  # not a plugin file
    if not isinstance(man, dict):
        raise TypeError(f"{path}: {attr} must be a dict, got {type(man).__name__}")
    if not isinstance(man.get("name"), str) or not man["name"]:
        raise ValueError(f"{path}: {attr}['name'] must be a non-empty string")
    if not callable(man.get("run")):
        raise TypeError(f"{path}: {attr}['run'] must be callable")
    return man
