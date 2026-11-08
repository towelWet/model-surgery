# Refuse merges unless complete configuration matches, excluding quantization metadata.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.surgery.checkpoint.configuration import read_configuration
import copy


def verify_configs(*sources):
    configs = [copy.deepcopy(read_configuration(src)) for src in sources]
    for cfg in configs:
        for section in (cfg, cfg.get("text_config", {})):
            section.pop("quantization", None)               # bit width may differ between sources
            section.pop("quantization_config", None)
    if any(cfg != configs[0] for cfg in configs[1:]):
        raise ValueError("checkpoint configurations differ beyond quantization; merge refused")
