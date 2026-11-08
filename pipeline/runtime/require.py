"""Require an adapter's explicit declaration; it is not independent verification."""
from .settings import runtime_settings


def require_capability(capability):
    if runtime_settings().get("capabilities", {}).get(capability) is not True:
        raise NotImplementedError(f"configured runtime has no enabled capability declaration: {capability}")
