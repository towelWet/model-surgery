"""Check protocol readiness without spawning a runtime, loading a model, or changing services."""
import json
import os
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline.runtime.command import runtime_command
from pipeline.runtime.endpoint import api_base_url
from pipeline.runtime.settings import runtime_settings


def main():
    failed = False
    for capability in sorted(runtime_settings().get("commands", {})):
        try:
            runtime_command(capability)
            print(f"PASS {capability} command available")
        except FileNotFoundError:
            print(f"SKIP optional {capability} command unavailable")
    profile = Path(__file__).resolve().parents[1] / "profile"
    for script in sorted((profile / "benchmark").glob("*.sh")):
        if not os.access(script, os.X_OK):
            print(f"FAIL {script.name} not executable")
            failed = True
    try:
        with urllib.request.urlopen(api_base_url() + "/models", timeout=3) as response:
            payload = json.load(response)
        if not isinstance(payload.get("data"), list) or not payload["data"]:
            raise ValueError("no model identities returned")
        print("PASS compatible models endpoint")
    except (OSError, ValueError):
        print("FAIL compatible models endpoint unavailable or invalid")
        failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
