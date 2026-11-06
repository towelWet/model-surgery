# _site_prompt.py — (system, user) messages for site-gen bench: free page, or fill a fixed scaffold.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _gen_task import REFERENCE_BRIEF, SCAFFOLD, SCAFFOLD_FILL_SYSTEM, SYSTEM_PROMPT


def site_prompt(scaffold):
    if not scaffold:
        return SYSTEM_PROMPT, REFERENCE_BRIEF               # model writes whole page
    user = (REFERENCE_BRIEF
            + "\n\nShell (already on disk — do not echo it):\n"
            + SCAFFOLD
            + "\n\nFill <!-- CONTENT --> only.")              # model writes body only; shell merged after
    return SCAFFOLD_FILL_SYSTEM, user
