"""Pass private backend environment names through configuration, without publishing them."""
import os
from .settings import runtime_settings


def runtime_environment():
    environment = os.environ.copy()
    environment.update({key: str(value) for key, value in runtime_settings().get("environment", {}).items()})
    return environment
