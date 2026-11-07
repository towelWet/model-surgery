"""Config decoder layer count selects the text stack beside vision/mtp stacks; ties still refuse."""
from pipeline.surgery._layer_prefix import decoder_layer_prefix


def test_config_layer_count_selects_text_stack_beside_vision_and_mtp():
    text = [f"language_model.model.layers.{i}.mlp.down_proj.weight" for i in range(4)]
    vision = [f"vision_tower.blocks.{i}.mlp.fc1.weight" for i in range(3)]
    mtp = ["mtp.layers.0.mlp.down_proj.weight"]                       # one-layer predictor stack
    assert decoder_layer_prefix(text + vision + mtp, 4) == "language_model.model.layers."
    for tied in ([*text, *[f"vision.blocks.{i}.w.weight" for i in range(4)]], text + vision):
        try:
            decoder_layer_prefix(tied, 4 if len(tied) == 8 else 5)    # equal depth, or count matches none
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous stacks accepted despite layer-count tie-break")
