# disabled.py — _hidden.capture is fail-closed without mlx_lm.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))            # project root for tests.*
from tests.profile.helpers.module import load                             # file-path loader + forbidden-module assert


def test_capture_raises_before_touching_model():
    mod = load("pipeline/profile/_hidden.py", "ms_hidden")
    try:
        mod.capture(None, None)                    # dummy args must never be dereferenced
    except NotImplementedError as e:
        assert "tensor-array" in str(e) and "residual" in str(e)
    else:
        raise AssertionError("capture() did not raise NotImplementedError")
