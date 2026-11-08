# _chat_row.py — wrap one user/assistant pair (+ shared system prompt) as a training row.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
SYSTEM = "You are an expert coding and reverse-engineering assistant."


def _row(user, assistant, tag):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "tag": tag,                                   # topic label: lets balanced builder filter rows
    }
