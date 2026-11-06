"""flaky_write: delegates to STATE['real'] once, then raises OSError (injected interruption)."""
from tests.storage.helpers.mutate import STATE


def flaky_write(path, data):
    STATE["calls"] += 1
    if STATE["calls"] > 1:
        raise OSError("injected interruption after first file")
    return STATE["real"](path, data)
