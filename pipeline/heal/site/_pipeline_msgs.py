# _pipeline_msgs.py — stage-1 site prompt (structure skeleton); re-exports stage 2 (polish).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "profile", "site"))  # site helpers (_gen_task, _gen_api …) moved here
from _gen_task import REFERENCE_BRIEF, SCAFFOLD

STAGE1_SYSTEM = """You output a minimal HTML skeleton only: semantic sections with placeholder text.
Single file, inline <style> with layout only (grid/flex, spacing). No images, no JS.
No markdown fences. Under 120 lines total."""

from _polish_msgs import STAGE2_SYSTEM, stage2_messages  # stage 2 split out; re-exported


def stage1_messages():
    return [
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": REFERENCE_BRIEF + "\nUse placeholders like <!-- hero -->. " + SCAFFOLD},
    ]

