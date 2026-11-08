# save/config.py — Write config.json; mirror mlx_lm save_config quantization_config alias.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os


def save_config(config: dict, dest_dir: str) -> None:
    """Write config.json; mirror mlx_lm save_config quantization_config alias."""
    cfg = dict(config)
    cfg.pop("_name_or_path", None)
    cfg.pop("vision_config", None)
    if "quantization" in cfg:
        cfg["quantization_config"] = cfg["quantization"]
    cfg = dict(sorted(cfg.items()))
    with open(os.path.join(dest_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
