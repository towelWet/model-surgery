# assistant_span.py — token range of final assistant message (positions the student is scored on).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from .apply_template import apply_template


def assistant_span(tok, messages, tools=None):
    """Return (full_ids, assistant_start, assistant_end) for teacher-forced positions."""
    full = messages
    if not full or full[-1].get("role") != "assistant":
        return None, None, None                            # nothing to distill
    prefix = full[:-1]
    full_ids = apply_template(tok, full, tools)
    pref_ids = apply_template(tok, prefix + [{"role": "assistant", "content": ""}], tools)
    # Positions to score: assistant tokens only (exclude final eos if duplicated).
    start = len(pref_ids)                                  # assumes empty-assistant render is a prefix of full
    end = len(full_ids)
    if end <= start:
        return None, None, None
    return full_ids, start, end
