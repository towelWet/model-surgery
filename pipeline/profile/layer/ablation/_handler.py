# ablation/_handler.py — HTTP handler: POST /v1/layer_ablation -> JSON NLL deltas.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
from http.server import BaseHTTPRequestHandler

from pipeline.profile.layer.ablation._hook import handle_layer_ablation


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") != "/v1/layer_ablation":    # only endpoint served
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")       # empty body -> {} -> hook raises "prompts required"
        data = json.dumps(handle_layer_ablation(body)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
