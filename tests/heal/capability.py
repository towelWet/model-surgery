"""A renamed executable cannot imply tool-schema training support."""
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from pipeline.runtime.require import require_capability


def test_structured_training_requires_explicit_adapter_contract():
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "runtime.toml"
        config.write_text('[runtime.capabilities]\nstructured_training = false\n')
        with patch.dict(os.environ, {"SURGERY_RUNTIME_CONFIG": str(config)}, clear=True):
            try:
                require_capability("structured_training")
            except NotImplementedError:
                pass
            else:
                raise AssertionError("unsupported structured training accepted")
            config.write_text(config.read_text().replace("false", "true"))
            require_capability("structured_training")
