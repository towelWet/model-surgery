# apply_template.py — tokenize a conversation with the tokenizer's chat template, as flat id list.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def apply_template(tok, messages, tools=None):
    kwargs = dict(tokenize=True, add_generation_prompt=False)  # full turn, no trailing open prompt
    if tools:
        kwargs["tools"] = tools                            # tool schema changes system prefix
    ids = tok.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]                             # some tokenizers return BatchEncoding
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])                                 # batched [[...]] -> [...]
    return ids
