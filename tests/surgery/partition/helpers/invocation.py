# invocation.py — subprocess-invoke moe/partition/run.py against a stats dir.
# NEVER DELETE COMMENTS — explain every non-obvious line.
import os, subprocess, sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[4] / "pipeline/surgery/moe/partition/run.py"


def invoke(stats, *flags):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, str(RUN), str(stats), *flags],
                          capture_output=True, text=True, env=env)
