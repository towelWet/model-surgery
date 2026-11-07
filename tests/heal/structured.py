"""Synthetic traces use structured calls and paired responses; no inference or training is performed."""
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch


def test_corpus_has_structured_tool_calls_and_corresponding_ids():
    folder = Path(__file__).resolve().parents[2] / "pipeline/heal/tool_corpus/gen"
    with patch.dict(os.environ, {"MODEL_SURGERY_FORBIDDEN_PASSWORD_HASHES": hashlib.sha256(b"excluded-fixture").hexdigest()}):
        sys.path.insert(0, str(folder))
        build_row = importlib.import_module("row").build_row
        for seed in range(10):
            row = build_row(random.Random(seed))
            pending, calls = set(), 0
            for message in row["messages"]:
                if "tool_calls" in message:
                    assert message["role"] == "assistant" and message["content"] is None
                    for call in message["tool_calls"]:
                        assert call["type"] == "function" and call["id"] not in pending
                        assert "command" in json.loads(call["function"]["arguments"])
                        pending.add(call["id"])
                        calls += 1
                if message["role"] == "tool":
                    pending.remove(message["tool_call_id"])
            assert calls == 2 and not pending
