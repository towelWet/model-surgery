"""The actual training entry refuses unsupported tool training before creating adapters or running commands."""
import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def test_training_capability_failure_has_no_side_effects():
    train = importlib.import_module("pipeline.heal.distill._train").train
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = root / "runtime.toml"
        config.write_text('[runtime.capabilities]\nstructured_training = false\n')
        data = root / "data.jsonl"
        data.write_text(json.dumps({"messages": [{"role": "assistant", "content": None,
            "tool_calls": [{"id": "fixture", "type": "function", "function": {
                "name": "bash", "arguments": "{}"}}]}],
            "tools": [{"type": "function", "function": {"name": "bash"}}]}) + "\n")
        args = SimpleNamespace(data=str(data), adapter_path=str(root / "adapter"))
        with patch.dict(os.environ, {"SURGERY_RUNTIME_CONFIG": str(config)}, clear=True):
            with patch("subprocess.run") as run:
                try:
                    train(args, str(root / "cache"))
                except NotImplementedError:
                    pass
                else:
                    raise AssertionError("unsupported tool training executed")
                run.assert_not_called()
        assert not (root / "adapter").exists()
