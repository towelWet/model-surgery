# disabled.py — layer/_tap.capture_mlp is fail-closed without mlx_lm.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))            # project root for tests.*
from tests.profile.helpers.module import load                             # file-path loader + forbidden-module assert


def test_capture_mlp_raises_before_touching_model():
    mod = load("pipeline/profile/layer/tap/_core.py", "ms_tap")
    calls = []
    try:
        mod.capture_mlp(None, None, calls.append, check=True)   # check path must fail first too
    except NotImplementedError as e:
        assert "tensor-array" in str(e) and "MLP" in str(e)
        assert not calls                                       # callback never fired
    else:
        raise AssertionError("capture_mlp() did not raise NotImplementedError")
