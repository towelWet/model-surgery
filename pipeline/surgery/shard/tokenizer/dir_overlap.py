"""Reject missing source, identical, or nested source/destination pairs."""


def check_dirs(source, destination):
    if not source.is_dir():
        raise FileNotFoundError(f"Missing tokenizer source: {source}")
    if source == destination:
        raise ValueError("Tokenizer source and destination are identical")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Tokenizer source and destination overlap")
