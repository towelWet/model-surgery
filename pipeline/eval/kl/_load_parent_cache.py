# _load_parent_cache.py — read parent traces; refuse if missing.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Old load_cache() silently BUILT the cache when missing — during a student run
# that recorded the student as "parent", so KL came out ~0 and looked perfect.
import json
import os

from _prompts import CACHE


def load_cache():
    """Return parent traces from CACHE; raise with fix instructions if absent."""
    if not os.path.exists(CACHE):
        raise SystemExit(f"no parent cache at {CACHE}\n"
                         "serve PARENT model, run: run.py --build-cache  (set MS_KL_CACHE per parent)")
    with open(CACHE) as f:
        return json.load(f)
