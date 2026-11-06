"""Shared mutable state for storage fault-injection seams (shard path, real fns, call count)."""
STATE = {"shard": None, "real": None, "calls": 0}
