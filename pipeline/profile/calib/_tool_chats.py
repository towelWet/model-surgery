# _tool_chats.py — CALIB prompts rendered WITH tool schema through model's chat template.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from calib._core import CALIB                 # fixed calibration prompts (no user data)
from _schema import TOOL_SCHEMA               # same tools god offers at runtime


def _tool_chats(tok):
    return [tok.apply_chat_template([{"role": "user", "content": p}], tools=TOOL_SCHEMA,
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False) for p in CALIB]
