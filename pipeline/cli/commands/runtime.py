"""Inspect runtime capabilities without printing private executable paths or backend aliases."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pipeline.runtime.command import runtime_command
from pipeline.runtime.settings import runtime_settings


def run(argv):
    if argv:
        raise ValueError("runtime takes no arguments")
    settings = runtime_settings()
    print("HTTP protocol: OpenAI-compatible")
    for capability in sorted(settings.get("commands", {})):
        try:
            runtime_command(capability)
            print(f"{capability}: available")
        except FileNotFoundError:
            print(f"{capability}: unavailable")
    return 0


COMMAND = {"name": "runtime", "help": "inspect configured runtime capabilities without private names", "run": run}
