"""TOML basic-string escaping via JSON serialization (escapes all C0 controls)."""
import json


def toml_str(value):
    return json.dumps(str(value), ensure_ascii=True)
