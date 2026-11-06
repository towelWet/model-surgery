# num_layers_from_config.py — Layer count from top-level or nested text_config.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def num_layers_from_config(config: dict) -> int:
    """Layer count from top-level or nested text_config."""
    tc = config.get("text_config", config)
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if key in tc:
            return int(tc[key])
    raise ValueError("config missing num_hidden_layers / n_layer")
