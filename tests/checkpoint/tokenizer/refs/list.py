"""Plural '_files' metadata keys contribute every listed path, list or dict."""
from pathlib import Path
from pipeline.surgery.shard.tokenizer.references import referenced_assets


def test_files_keys_collect_list_and_dict_refs():
    metadata = {
        "vocab_files": ["a.lex", "b.lex"],                    # list form
        "aux_files": {"main": "x.lex", "missing": None},      # dict form; null ignored
    }
    assert referenced_assets(metadata) == {Path("a.lex"), Path("b.lex"), Path("x.lex")}
