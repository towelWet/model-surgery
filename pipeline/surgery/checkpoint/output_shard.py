"""Choose a shard name that cannot collide with preserved assets or earlier output shards."""


def output_shard(number, occupied):
    while True:
        name = f"model-{number:05d}.safetensors"
        if not any(path == name or path.startswith(name + "/") for path in occupied):
            occupied.add(name)
            return name
        number += 1
