"""HTTP configuration is reloaded per call; private environment alias names come from config."""
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from pipeline.runtime.endpoint import api_base_url
from pipeline.runtime.settings import runtime_settings


def test_runtime_http_endpoint_and_aliases_are_dynamic():
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "private.toml"
        config.write_text('[runtime.http]\nbase_url = "http://localhost:{port}/v1"\nport = 9011\n'
                          '[runtime.aliases]\nmodel_path = ["CUSTOM_MODEL_VARIABLE"]\n')
        env = {"SURGERY_RUNTIME_CONFIG": str(config), "CUSTOM_MODEL_VARIABLE": "/model/fixture"}
        with patch.dict(os.environ, env, clear=True):
            assert api_base_url() == "http://localhost:9011/v1"
            assert runtime_settings()["http"]["model_path"] == "/model/fixture"
            config.write_text(config.read_text().replace("9011", "9012"))
            assert api_base_url() == "http://localhost:9012/v1"
            with patch.dict(os.environ, {"SURGERY_API_URL": "http://localhost:7777/custom/v1"}):
                assert api_base_url() == "http://localhost:7777/custom/v1"
