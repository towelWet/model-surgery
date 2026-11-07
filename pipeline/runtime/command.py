"""Resolve a capability's configured executable candidates; no backend-name branching."""
import os
import shlex
import shutil
from .settings import runtime_settings


def runtime_command(capability):
    settings = runtime_settings()
    candidates = settings.get("commands", {}).get(capability, {}).get("candidates", [])
    for name in settings.get("command_environment", {}).get(capability, []):
        if os.environ.get(name):
            candidates = [shlex.split(os.environ[name])]
            break
    for candidate in candidates:
        if not isinstance(candidate, list) or not candidate or any(type(v) is not str for v in candidate):
            raise ValueError(f"invalid configured command for capability: {capability}")
        executable = shutil.which(os.path.expanduser(candidate[0]))
        if executable:
            return [executable, *candidate[1:]]
    raise FileNotFoundError(f"no executable configured or available for runtime capability: {capability}")
