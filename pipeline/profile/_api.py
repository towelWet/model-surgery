# _api.py — HTTP POST helper for tensor-array OpenAI-compatible API on :8765.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline.runtime.endpoint import api_base_url
from pipeline.runtime.settings import runtime_settings
import urllib.request

API = api_base_url()  # compatibility export; requests re-read current configuration


def _post(endpoint, body_dict):
    """POST JSON body to endpoint; return parsed response dict."""
    body = json.dumps(body_dict).encode()
    req = urllib.request.Request(
        f"{api_base_url()}/{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    # 180s cap: hung Metal or dead server fails fast instead of blocking hours
    return json.loads(urllib.request.urlopen(req, timeout=float(runtime_settings()["http"]["timeout_seconds"])).read())
