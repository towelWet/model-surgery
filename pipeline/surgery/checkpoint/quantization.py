"""Resolve explicit, per-module MLX affine quantization settings; never guess packing."""


def quantization_settings(cfg, key):
    prefix = key.removesuffix(".weight")
    text = cfg.get("text_config", cfg)
    quant = cfg.get("quantization") or text.get("quantization")
    if quant is None:
        raise ValueError(f"{key}: packed tensor requires explicit quantization metadata")
    if not isinstance(quant, dict):
        raise ValueError(f"{key}: malformed quantization metadata")
    settings = {k: v for k, v in quant.items() if not isinstance(v, dict)}
    override = quant.get(prefix, {})
    if not isinstance(override, dict):
        raise ValueError(f"{key}: incompatible per-module quantization override")
    settings.update(override)
    bits, group = settings.get("bits"), settings.get("group_size")
    if type(bits) is not int or type(group) is not int or bits <= 0 or group <= 0:
        raise ValueError(f"{key}: explicit positive integer bits and group_size required")
    if settings.get("mode", "affine") != "affine":
        raise ValueError(f"{key}: only MLX affine quantization is supported")
    return {"bits": bits, "group_size": group}
