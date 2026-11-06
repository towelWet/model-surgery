# _api.py — HTTP POST helper for tensor-array OpenAI-compatible API on :8765.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os
import urllib.request

_port = os.environ.get("TARR_EVAL_PORT", "8765")        # override with env var
API = f"http://localhost:{_port}/v1"                     # tensor-array API base


def _post(endpoint, body_dict):
    """POST JSON body to endpoint; return parsed response dict."""
    body = json.dumps(body_dict).encode()
    req = urllib.request.Request(
        f"{API}/{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    # 180s cap: hung Metal or dead server fails fast instead of blocking hours
    return json.loads(urllib.request.urlopen(req, timeout=180).read())
