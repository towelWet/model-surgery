# moe/_target.py — dense source architecture -> MoE layout to emit (block name + config writer).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Both targets store router at <block>.gate and stacked experts at <block>.switch_mlp.* (MLX
# convention); runtime picks prefix from weights (qwen/3/moe/mlp/moe-prefix.cpp).
# Arch table moved to targets/*.toml (was hardcoded TARGETS dict); this file only resolves.
import importlib
from pipeline.surgery.moefy._target_packs import target_packs


def moe_target(cfg):
    arch = (cfg.get("architectures") or ["?"])[0]
    pack = target_packs().get(arch)
    if pack is None:
        raise ValueError(f"no dense->MoE target pack for {arch} (add pipeline/surgery/moefy/targets/*.toml)")
    if pack["converter"] != "pipeline/surgery/moefy":           # repo-relative converter dir (TOML)
        raise ValueError(f"{arch}: converted by {pack['converter']}/, not pipeline/surgery/moefy/")
    module, func = pack["config_writer"].split(":")            # absolute "pipeline.x.y:function" (TOML)
    return pack["block"], getattr(importlib.import_module(module), func)  # writer lives beside converter
