# _merge_scaffold.py — put model's body fragment into the site scaffold (full docs pass through).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def merge_scaffold(fragment: str, scaffold: str, marker: str) -> str:
    """Insert model body fragment into scaffold; if model sent a full doc, return as-is."""
    raw = (fragment or "").strip()
    low = raw.lower()
    if low.startswith("<!doctype") or "<html" in low[:200]:
        return raw                                          # model ignored the shell: keep its page
    return scaffold.replace(marker, raw, 1)
