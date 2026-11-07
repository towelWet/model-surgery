"""Renaming a trainer requires only private config; wrappers and environment aliases remain functional."""
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
from pipeline.runtime.command import runtime_command
from pipeline.runtime.environment import runtime_environment


def test_runtime_commands_are_independent_of_backend_brand():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        program = folder / "renamed engine"
        program.write_text('#!/bin/sh\nprintf "%s" "$CUSTOM_RUNTIME_ALIAS"\n')
        program.chmod(0o755)
        config = folder / "private.toml"
        config.write_text('[runtime.commands.train]\ncandidates = [["missing-command"], ["' +
                          str(program) + '"]]\n[runtime.environment]\nCUSTOM_RUNTIME_ALIAS = "works"\n')
        with patch.dict(os.environ, {"SURGERY_RUNTIME_CONFIG": str(config)}, clear=True):
            command = runtime_command("train")
            assert command == [str(program)]
            result = subprocess.run(command, env=runtime_environment(), capture_output=True, text=True)
            assert result.returncode == 0 and result.stdout == "works"
            renamed = folder / "another engine"
            program.rename(renamed)
            config.write_text(config.read_text().replace(str(program), str(renamed)))
            assert runtime_command("train") == [str(renamed)]
