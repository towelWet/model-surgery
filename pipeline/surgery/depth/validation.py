"""Validate depth-removal policy before creating an output checkpoint."""
from pipeline.surgery.checkpoint.destination import validate_destination


def validate_depth(config, removed, destination):
    text = config.get("text_config", config)
    count = text.get("num_hidden_layers")
    if type(count) is not int or count < 1:
        raise ValueError("config needs a positive num_hidden_layers")
    if not removed or any(type(i) is not int or i < 0 or i >= count for i in removed):
        raise ValueError(f"removed layers must be a nonempty subset of 0..{count - 1}")
    if len(removed) >= count:
        raise ValueError("cannot remove every decoder layer")
    validate_destination(destination)                        # sources bound at CLI call site
    if "layer_types" in text and len(text["layer_types"]) != count:
        raise ValueError("layer_types count disagrees with num_hidden_layers")
    text["num_hidden_layers"] = count - len(removed)
    if "layer_types" in text:
        text["layer_types"] = [kind for i, kind in enumerate(text["layer_types"]) if i not in removed]
    return config
