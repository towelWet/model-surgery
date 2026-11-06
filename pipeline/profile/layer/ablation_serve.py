# ablation_serve.py — dev stub: POST /v1/layer_ablation on PORT (default 8765).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Port 8765 = tensor-array default; set PORT to avoid clashing with a live server.
import os
from http.server import HTTPServer

from _ablation_handler import Handler


def main():
    port = int(os.environ.get("PORT", "8765"))
    HTTPServer(("", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
