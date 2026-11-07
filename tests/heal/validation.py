"""Training refuses historical textual tool-call records and incomplete tool responses."""
import json
import tempfile
from pathlib import Path
from pipeline.heal.distill.validate_data import validate_training_data


def test_training_data_rejects_legacy_serialization():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "corpus.jsonl"
        bad = [{"messages": [{"role": "assistant", "content": json.dumps([
                    {"function": {"name": "bash", "arguments": "{}"}}])}]},
               {"messages": [{"role": "tool", "content": "orphan"}]},
               {"messages": [{"role": "assistant", "content": '<tool_call>\n{"name": "bash"}\n</tool_call>'}]},
               {"teacher_fill": True, "messages": [{"role": "assistant", "content": "placeholder"}]}]
        for row in bad:
            path.write_text(json.dumps(row) + "\n")
            try:
                validate_training_data(path)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid training row accepted")
        row = {"messages": [{"role": "assistant", "content": None, "tool_calls": [
            {"id": "one", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "one", "content": "done"},
            {"role": "assistant", "content": "complete"}]}
        path.write_text(json.dumps(row) + "\n")
        assert validate_training_data(path) == {"rows": 1, "structured_calls": 1}
