# _html_unfence.py — strip leading/trailing ``` fences from model HTML output.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Differs from eval/_strip_fences.py on purpose: tolerates a missing closing fence
# (site replies often truncate at max_tokens), so they are not merged.
import re


def strip_fences(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)  # opening fence + optional lang tag
        t = re.sub(r"\n?```\s*$", "", t)        # closing fence if present
    return t.strip()
