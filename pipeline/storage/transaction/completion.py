"""Require a present complete.toml whose recorded hash matches manifest.toml."""
import tomllib
from ..hashes.digest import file_sha256


def check_complete(store):
    marker = store / "complete.toml"
    if not marker.is_file():
        raise ValueError("store incomplete: no complete.toml marker")
    recorded = tomllib.loads(marker.read_text()).get("manifest_sha256")
    if recorded != file_sha256(store / "manifest.toml"):
        raise ValueError("complete.toml does not match manifest.toml")
    return True
