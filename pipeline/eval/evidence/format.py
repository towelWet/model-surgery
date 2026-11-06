"""Encode scalar TOML values without losing precision or string escaping."""
import json
import math


def format_scalar(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if type(value) is bool:
        return str(value).lower()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("report values must be finite scalar numbers, strings, or booleans")
    return repr(value)
