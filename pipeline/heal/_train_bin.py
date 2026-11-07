# Compatibility shim: resolve the configured trainer without depending on its product name.
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline.runtime.command import runtime_command


def train_bin():
    command = runtime_command("train")
    if len(command) != 1:
        raise ValueError("train_bin requires a single executable; use runtime_command for wrapper arguments")
    return command[0]
