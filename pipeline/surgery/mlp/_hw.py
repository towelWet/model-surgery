# _hw.py — hidden / intermediate sizes from text_config (Tau 27B or Empero 9B).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def mlp_hw(text_config):
    """Return (hidden_size, intermediate_size) from text_config or a flat config."""
    return int(text_config["hidden_size"]), int(text_config["intermediate_size"])
