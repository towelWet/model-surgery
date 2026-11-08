"""Real tokenizers BPE: trimmed tokenizer spells corpus text with same tokens, other text still encodes."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import sys
from pathlib import Path
from tokenizers import Tokenizer, models


def test_trimmed_tokenizer_roundtrip():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.vocab._bpe_closure import bpe_closure
    from pipeline.surgery.vocab._trim_tokenizer import trim_tokenizer
    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4, "bc": 5, "ca": 6}
    merges = [("a", "b"), ("ab", "c"), ("b", "c"), ("c", "a")]
    tj = json.loads(Tokenizer(models.BPE(vocab, merges)).to_str())
    full = Tokenizer.from_str(json.dumps(tj))
    kept = bpe_closure(tj, set(full.encode("abcabc").ids))
    small = Tokenizer.from_str(json.dumps(trim_tokenizer(tj, {o: n for n, o in enumerate(kept)})))
    assert small.encode("abcabc").tokens == full.encode("abcabc").tokens   # corpus text unchanged
    assert full.encode("bcca").tokens == ["bc", "ca"]
    assert small.encode("bcca").tokens == ["b", "c", "c", "a"]             # dropped merges -> alphabet
