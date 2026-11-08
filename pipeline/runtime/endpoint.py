"""Resolve an OpenAI-compatible endpoint from runtime configuration at request time."""
from urllib.parse import urlsplit
from .settings import runtime_settings


def api_base_url():
    settings = runtime_settings()["http"]
    url = settings["base_url"].format_map(settings).rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("runtime base URL must be HTTP(S) without embedded credentials")
    return url
