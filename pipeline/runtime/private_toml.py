# private_toml.py — where model-surgery's private values live (private-toml system).
# Why: real workstation paths/aliases/hashes must never sit inside the published tree. They live in
#      $CLAUDE_DIR/private/model-surgery.toml (shape shipped as local.example.toml);
#      SURGERY_RUNTIME_CONFIG overrides for tests or other machines.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
from pathlib import Path


def private_toml_path():
    if os.environ.get("SURGERY_RUNTIME_CONFIG"):
        return Path(os.environ["SURGERY_RUNTIME_CONFIG"])                    # explicit override wins
    claude = Path(os.environ.get("CLAUDE_DIR", Path.home() / ".claude"))    # no hardcoded home
    return claude / "private" / "model-surgery.toml"
