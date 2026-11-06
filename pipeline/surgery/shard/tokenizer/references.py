"""Read local file references declared by checkpoint tokenizer metadata."""
from .ref_collect import collect_refs


def referenced_assets(value):
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            collect_refs(result, key, item)
            if isinstance(item, (dict, list)):
                result.update(referenced_assets(item))         # nested objects still scanned
    elif isinstance(value, list):
        for item in value:
            result.update(referenced_assets(item))
    return result
