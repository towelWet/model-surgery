"""Discover unfamiliar indexed block paths and refuse ambiguous text/vision stacks."""
from pipeline.surgery._layer_prefix import decoder_layer_prefix


def test_layer_prefix_discovery_is_not_architecture_list():
    keys = ["unfamiliar.decoder.blocks.0.ffn.weight", "unfamiliar.decoder.blocks.1.attention.weight"]
    assert decoder_layer_prefix(keys) == "unfamiliar.decoder.blocks."
    for ambiguous in ([], [*keys, "vision.blocks.0.attention.weight"]):
        try:
            decoder_layer_prefix(ambiguous)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous or missing block namespace accepted")
