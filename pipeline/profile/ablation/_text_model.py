# _text_model.py — find text decoder in a (possibly VLM-wrapped) model object.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def _text_model(model):
    tm = getattr(model, "language_model", model)    # VLM wrapper -> its LM; plain LM -> itself
    inner = tm.model if hasattr(tm, "model") else tm  # decoder stack (embed/layers/norm)
    return tm, inner
