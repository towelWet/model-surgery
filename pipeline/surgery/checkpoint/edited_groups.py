"""Stream original tensor groups with a bounded set of edited components substituted."""
from .tensor_groups import tensor_groups


def edited_groups(source, mapping, edited):
    for group in tensor_groups(source, mapping, {}):
        group.update({key: edited[key] for key in group if key in edited})
        yield group
