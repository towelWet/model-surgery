"""Explicit model reload through the selected backend's own command; starts no background service."""
import argparse
import os
import subprocess
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline.runtime.command import runtime_command
from pipeline.runtime.environment import runtime_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    command = [part.format(model=args.model) for part in runtime_command("reload")]
    subprocess.run(command, env=runtime_environment(), check=True)


if __name__ == "__main__":
    main()
