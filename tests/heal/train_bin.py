"""train_bin resolves env first, then PATH; clear error when neither exists."""
import os
import sys
from pathlib import Path
from unittest.mock import patch


def test_train_bin_env_then_which_then_error():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline/heal"))
    from _train_bin import train_bin
    with patch.dict(os.environ, {"TENSOR_ARRAY_TRAIN": "/env/bin/tat"}):
        assert train_bin() == "/env/bin/tat"                 # env wins over PATH
    with patch.dict(os.environ, {}, clear=True):             # env var provably absent
        with patch("shutil.which", return_value="/path/bin/tensor-array-train"):
            assert train_bin() == "/path/bin/tensor-array-train"
        with patch("shutil.which", return_value=None):
            try:
                train_bin()
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("missing binary accepted")
