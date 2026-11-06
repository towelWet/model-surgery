# disabled.py — ablation/_forward.forward_logits is fail-closed without mlx_lm.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))            # project root for tests.*
from tests.profile.helpers.module import load                             # file-path loader + forbidden-module assert


def test_forward_logits_raises_before_touching_model():
    mod = load("pipeline/profile/ablation/_forward.py", "ms_forward")
    try:
        mod.forward_logits(None, None, skip_layers=[3])          # dummy args never dereferenced
    except NotImplementedError as e:
        assert "tensor-array" in str(e) and "layer-skip" in str(e)
    else:
        raise AssertionError("forward_logits() did not raise NotImplementedError")
