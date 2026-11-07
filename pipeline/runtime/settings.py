"""Hot-load runtime configuration; private paths/aliases never belong in public code."""
import os
from pathlib import Path
import tomllib
from .merge import merge_settings


def runtime_settings():
    defaults = Path(__file__).with_name("defaults.toml")
    result = tomllib.loads(defaults.read_text())
    local = Path(os.environ.get("SURGERY_RUNTIME_CONFIG", defaults.parents[2] / "local.toml"))
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
