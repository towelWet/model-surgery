# token_id.py — map a logprob token string back to a vocab id (None if it re-tokenizes to >1 id).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Lossy by nature: strings that don't round-trip to a single id are dropped by the caller.


def token_id(tok, piece):
    if piece == "":
        return tok.eos_token_id                            # server reports EOS as empty string
    ids = tok(piece, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None
