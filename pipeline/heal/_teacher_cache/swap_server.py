# swap_server.py — ask tensor-array to serve another checkpoint via profile/swap_model.sh.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import subprocess


def swap_server(model_path):
    swap = os.path.join(os.path.dirname(__file__), "..", "..", "profile", "swap_model.sh")  # pipeline/profile/
    subprocess.run(["bash", swap, model_path], check=True)  # raises if swap fails: never cache wrong model
