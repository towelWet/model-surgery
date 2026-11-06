# _validators.py — argument validation helpers for tool-call evaluation.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json


def is_valid_json_args(args_str, required_key):
    """Return True if args_str is valid JSON and contains required_key with non-empty value."""
    try:
        d = json.loads(args_str)
        return required_key in d and bool(d[required_key])  # key must exist and be truthy
    except Exception:
        return False                         # malformed JSON or missing key = invalid
