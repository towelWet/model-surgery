"""Invalid removals must never create output or corrupt declared layer counts."""
import tempfile
from pathlib import Path
from pipeline.surgery.depth.validation import validate_depth


def test_depth_preflight_rejects_invalid_policy_without_output():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "child-Tau"                        # dest names must include Tau
        for removed in (set(), {-1}, {4}, {0, 1, 2, 3}):
            try:
                validate_depth({"num_hidden_layers": 4}, removed, output)
            except ValueError:
                pass
            else:
                raise AssertionError(removed)
            assert not output.exists()
        config = validate_depth({"text_config": {"num_hidden_layers": 4}}, {1}, output)
        assert config["text_config"]["num_hidden_layers"] == 3
        config = validate_depth({"num_hidden_layers": 3, "layer_types": ["a", "b", "c"]}, {1}, output)
        assert config["layer_types"] == ["a", "c"]
        output.mkdir()
        try:
            validate_depth({"num_hidden_layers": 4}, {1}, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing destination accepted")
