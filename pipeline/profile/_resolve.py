# _resolve.py — unwrap Model -> TextModel -> Qwen3_5TextModel.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def _resolve(model):
    """Unwrap Model -> TextModel -> Qwen3_5TextModel.

    qwen3_5.Model (ForConditionalGeneration) holds .language_model (TextModel),
    which holds .model (Qwen3_5TextModel) and owns make_cache(). Checkpoints
    saved as TextOnly may load as TextModel directly, so fall back to model.
    """
    tm = getattr(model, "language_model", model)
    return tm.model, tm
