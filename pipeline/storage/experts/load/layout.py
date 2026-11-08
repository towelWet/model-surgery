"""Merge every layouts/*.toml into one grammar dict (lists union, scalars must agree).
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
"""
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "layouts"


def load_layout(folder=HERE):
    """Return merged layout; new arch spellings = new TOML file, zero code edits."""
    out, src = {}, {}
    for path in sorted(Path(folder).glob("*.toml")):
        for key, val in tomllib.loads(path.read_text()).items():
            if isinstance(val, list):
                out[key] = list(dict.fromkeys(out.get(key, []) + val))   # ordered union
            elif key in out and out[key] != val:
                raise ValueError(f"layout key {key!r}: {src[key]} says {out[key]!r}, {path.name} says {val!r}")
            else:
                out[key], src[key] = val, path.name
    if not out:
        raise ValueError(f"{folder}: no layout *.toml found")
    return out
