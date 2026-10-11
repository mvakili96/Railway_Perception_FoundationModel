"""Stdlib composite configuration and safe SAM-only checkpoint selection."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from model.qwen_sam_config import QwenSAMConfig, select_sam_state_dict


class CompositeConfigTest(unittest.TestCase):
    def test_defaults_match_agreed_model_not_training_entrypoint(self):
        config = QwenSAMConfig()
        self.assertEqual(config.seg_prompt_bridge_type, "single")
        self.assertEqual(config.sam_model_type, "vit_h")
        self.assertEqual(config.train_sam_last_blocks, 16)
        self.assertTrue(config.train_sam_prompt_encoder)
        self.assertIsNone(config.sam_init_source)

    def test_config_round_trip_preserves_token_capacity_and_provenance(self):
        config = replace(QwenSAMConfig(), seg_token_id=151669, qwen_vocab_capacity=151936,
            source_qwen_revision="a" * 40, sam_init_source="rail_checkpoint",
            sam_init_path="/checkpoint/rail", seg_prompt_bridge_type="four_query")
        with tempfile.TemporaryDirectory() as temp:
            config.save_pretrained(temp)
            self.assertEqual(QwenSAMConfig.from_pretrained(temp), config)

    def test_invalid_configs_rejected(self):
        for changed in ({"seg_prompt_bridge_type": "other"}, {"seg_query_num_heads": 3},
                        {"sam_model_type": "vit_b"}, {"schema_version": 2},
                        {"train_sam_last_blocks": -1}, {"seg_token_id": -1},
                        {"seg_token_id": 10, "qwen_vocab_capacity": 10}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                replace(QwenSAMConfig(), **changed)

    def test_original_sam_complete_exact(self):
        state = {"image_encoder.weight": "image", "mask_decoder.weight": "mask"}
        self.assertEqual(select_sam_state_dict(state, state.keys(), "pretrained"), state)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            select_sam_state_dict({**state, "llama.weight": "language"}, state.keys(), "pretrained")

    def test_rail_checkpoint_selects_only_sam_and_strips_prefixes(self):
        state = {"module.base_model.model.model.visual_model.image_encoder.weight": "image",
                 "module.base_model.model.model.visual_model.mask_decoder.weight": "mask",
                 "module.base_model.model.model.embed_tokens.weight": "DO NOT TRANSFER",
                 "module.base_model.model.model.text_hidden_fcs.weight": "DO NOT TRANSFER"}
        self.assertEqual(select_sam_state_dict({"module": state},
            ["image_encoder.weight", "mask_decoder.weight"], "rail_checkpoint"),
            {"image_encoder.weight": "image", "mask_decoder.weight": "mask"})

    def test_partial_duplicate_and_partitioned_sam_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_sam_state_dict({"model.visual_model.mask_decoder.weight": "mask"},
                                 ["image_encoder.weight", "mask_decoder.weight"], "rail_checkpoint")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_sam_state_dict({"sam.weight": 1, "model.visual_model.weight": 2}, ["weight"], "rail_checkpoint")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_sam_state_dict({"optimizer_state_dict": {}}, ["weight"], "rail_checkpoint")
        with self.assertRaisesRegex(ValueError, "explicit"):
            select_sam_state_dict({}, [], None)

    def test_new_model_has_no_legacy_llava_or_clip_import(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "model/QwenSAM.py").read_text()
        self.assertNotIn("from model.LISA", source)
        self.assertNotIn("from .llava", source)
        self.assertNotIn("CLIPVisionModel", source)


if __name__ == "__main__":
    unittest.main()
