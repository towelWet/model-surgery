# _layer_prefix.py — decoder layer key prefixes (Tau vs Empero nested VLM).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

# Tau-56L / Qwen3.8-27B MLX uses language_model.model.layers.
# Empero Qwen3.5-9B HF uses model.language_model.layers. (vision-language nest).
# model.layers. covers text-only HF dumps. MTP keys (mtp.layers.) are not decoders.
DECODER_PREFIXES = (
    "language_model.model.layers.",
    "model.language_model.layers.",
    "model.layers.",
)


def decoder_layer_prefix(keys):
    """Return the decoder layer prefix present in keys, or raise."""
    for prefix in DECODER_PREFIXES:
        if any(k.startswith(prefix) for k in keys):
            return prefix
    raise ValueError("no decoder layer prefix in weight keys")
