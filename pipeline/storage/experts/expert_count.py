"""Read routed-expert count E from config.json using layout keys; refuse dense configs (no guessing).
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
"""
import json
from pathlib import Path


def expert_count(src, lay):
    """Return E (int > 1) from src/config.json, top level first, then each configured nested scope."""
    cfg = json.loads((Path(src) / "config.json").read_text())
    scopes = [cfg] + [cfg.get(s) or {} for s in lay["config_scopes"]]
    for scope in scopes:
        for key in lay["expert_count_keys"]:
            if isinstance(scope.get(key), int) and scope[key] > 1:
                return scope[key]
    raise ValueError(f"{src}: config.json names no routed experts (dense? use `ms moeify` first)")
