# layer.py — split strategy "layer": every tensor of G consecutive layers (attention, router, experts) together.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Layer-less tensors (embeddings, final norm, lm_head) go to shared_file.


def file_of(info, group, lay):
    """Layer L -> layer_file[L // G]; no layer index -> shared_file."""
    if info["layer"] is None:
        return lay["shared_file"]                                  # global tensors outside the block stack
    return lay["layer_file"].format(index=info["layer"] // group)


STRATEGY = {"name": "layer", "help": "one file per G layers; layer-by-layer execution",
            "run": file_of}                                        # run = file_of (plugin contract key)
