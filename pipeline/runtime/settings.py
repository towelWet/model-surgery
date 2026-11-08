"""Hot-load runtime configuration; private paths/aliases never belong in public code."""
import os
from pathlib import Path
import tomllib
from .merge import merge_settings
from .private_toml import private_toml_path


def runtime_settings():
    defaults = Path(__file__).with_name("defaults.toml")
    result = tomllib.loads(defaults.read_text())
    local = private_toml_path()  # $CLAUDE_DIR/private/model-surgery.toml (private-toml system)
    if local.is_file():
        result = merge_settings(result, tomllib.loads(local.read_text()).get("runtime", {}))
    elif "SURGERY_RUNTIME_CONFIG" in os.environ:
        raise FileNotFoundError("configured runtime settings file is missing")
    for key, names in result.get("aliases", {}).items():
        for name in names:
            if os.environ.get(name):
                result["http"][key] = os.environ[name]
                break
    return result
