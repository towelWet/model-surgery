"""object_pairs_hook that rejects duplicate JSON object keys."""


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in safetensors header: {key}")
        result[key] = value
    return result
