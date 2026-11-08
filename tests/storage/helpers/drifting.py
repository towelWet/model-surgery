"""drifting_region: wraps real region_sha256, appends bytes to STATE['shard'] mid-plan."""
from tests.storage.helpers.mutate import STATE


def drifting_region(path, start, length):
    digest = STATE["real"](path, start, length)
    with open(STATE["shard"], "ab") as stream:              # source grows mid-plan
        stream.write(b"drift")
    return digest
