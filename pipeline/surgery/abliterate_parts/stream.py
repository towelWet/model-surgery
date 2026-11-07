"""Project one output matrix at a time, preserving its original quantization contract."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from checkpoint.tensor_groups import tensor_groups
from checkpoint.read_weight import read_weight
from checkpoint.write_weight import encode_weight
from projection.apply import project_out
from _data import TARGET_ENDS


def projected_groups(source, mapping, config, direction, alpha):
    cache = {}
    for group in tensor_groups(source, mapping, cache):
        for key in tuple(group):
            if key.endswith(TARGET_ENDS):
                dense, options, dtype = read_weight(source, key, mapping, cache, config)
                group.update(encode_weight(key, project_out(dense, direction, alpha), options, dtype))
        yield group
