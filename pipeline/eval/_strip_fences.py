# _strip_fences.py — drop one wrapping markdown code fence before grading.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import re


def strip_fences(text):
    """Remove ```lang ... ``` wrapper so substring grading sees body only."""
    if not text:
        return ""
    t = text.strip()
    m = re.match(r"^```[\w]*\n(.*)```\s*$", t, re.DOTALL | re.IGNORECASE)  # whole reply is one fence
    return (m.group(1) if m else t).strip()
