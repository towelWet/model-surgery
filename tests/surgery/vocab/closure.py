"""BPE closure keeps alphabet, added tokens and merge parents; trim renumbers and drops merges."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path


TINY = {"model": {"vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4, "bc": 5},
                  "merges": [["a", "b"], ["ab", "c"], ["b", "c"]]},
        "added_tokens": [{"id": 6, "content": "<eos>"}]}


def test_closure_and_trim():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.vocab._bpe_closure import bpe_closure
    from pipeline.surgery.vocab._trim_tokenizer import trim_tokenizer
    from pipeline.surgery.vocab._remap_token_ids import remap_token_ids
    tj = TINY                                                 # trim deep-copies; constant stays intact
    kept = bpe_closure(tj, {4})
    assert kept == [0, 1, 2, 3, 4, 6]                         # "abc" pulls "ab"; "bc" unused -> dropped
    remap = {old: new for new, old in enumerate(kept)}
    out = trim_tokenizer(tj, remap)
    assert out["model"]["vocab"] == {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}
    assert out["model"]["merges"] == [["a", "b"], ["ab", "c"]]  # b+c merge gone with its result
    assert out["added_tokens"][0]["id"] == 5
    cfg = remap_token_ids({"eos_token_id": [6, 0], "bos_token_id": 6, "x": 1}, remap)
    assert cfg == {"eos_token_id": [5, 0], "bos_token_id": 5, "x": 1}
