# Refuse merges unless complete configuration matches, excluding quantization metadata.
from checkpoint.configuration import read_configuration
import copy


def verify_configs(src_a, src_b):
    configs = [copy.deepcopy(read_configuration(src)) for src in (src_a, src_b)]
    for cfg in configs:
        for section in (cfg, cfg.get("text_config", {})):
            section.pop("quantization", None)
            section.pop("quantization_config", None)
    if configs[0] != configs[1]:
        raise ValueError("checkpoint configurations differ beyond quantization; merge refused")
