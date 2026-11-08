# _polish_msgs.py — stage-2 site prompt: polish a draft skeleton into a finished page.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # pipeline/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile.site.gen._task import SYSTEM_PROMPT                 # shared site-gen system prompt

STAGE2_SYSTEM = SYSTEM_PROMPT + "\nYou will receive a draft HTML skeleton. Polish typography, colors, and copy only."


def stage2_messages(draft_html):
    return [
        {"role": "system", "content": STAGE2_SYSTEM},
        {"role": "user", "content": "Polish this draft into a finished page:\n\n" + draft_html},
    ]
