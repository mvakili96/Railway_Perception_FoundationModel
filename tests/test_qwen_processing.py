"""Native Stage 2 gate: CPU only, no Qwen/SAM weights or dataset mutations.

Set QWEN_TEST_MODEL_DIR to the downloaded snapshot to exercise its real processor.
Missing dependencies/snapshot skip locally; qwen_data_check rejects every skip.
"""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ML_ERROR = None
try:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from utils.dataset import HybridDataset, ValDataset
    from utils.reason_seg_dataset import ReasonSegDataset
    from utils.sem_seg_dataset import SemSegDataset
    from utils.qwen_processing import QwenProcessingConfig, QwenSegCollator
    from utils.qwen_token_alignment import decode_exact
    from utils.rail_reasoning import RAIL_REASONING_DECISION_PATTERN
    from utils.segmentation_sample import StructuredSegSample
except ModuleNotFoundError as exc:
    ML_ERROR = str(exc)

EXPLANATION = (
    "This is a turnout switch. The right blade is open and the left blade is closed. "
    "The open right blade and the closed left blade together create a continuous rail "
    "connection toward the right-hand path and break continuity with the left-hand path. "
    "Therefore, the ego-path follows the right-hand path."
)
MASK_ANSWER = "The segmentation result is [SEG]."
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(ML_ERROR, "Stage 2 ML dependencies unavailable: " + str(ML_ERROR))
class DatasetAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rgb = np.zeros((48, 96, 3), dtype=np.uint8)
        rgb[:, :32] = [230, 10, 25]
        rgb[:, 32:] = [5, 180, 220]
        annotation = {"text": ["Which is the right-hand path?", "Select the right route."],
                      "is_sentence": True, "shapes": [{"label": "route",
                      "points": [[1, 1], [25, 1], [25, 45], [1, 45]]}]}
        for dataset, split in (("ReasonSegRail", "train"), ("ReasonSeg", "val")):
            directory = self.root / "reason_seg" / dataset / split
            directory.mkdir(parents=True)
            Image.fromarray(rgb).save(directory / "rs08006.jpg")
            (directory / "rs08006.json").write_text(json.dumps(annotation))
        rail = self.root / "reason_seg/ReasonSegRail"
        (rail / "explanatory").mkdir()
        (rail / "explanatory/train.json").write_text(json.dumps([{
            "image": "rs08006.jpg", "query": annotation["text"][0], "outputs": EXPLANATION,
        }]))
        (rail / "weight_maps").mkdir()
        weights = np.zeros((48, 96), dtype=np.uint8)
        weights[:, :32] = 255
        Image.fromarray(weights).save(rail / "weight_maps/rs08006_weight.png")
        sem = self.root / "RailSem19-SemSeg-LISA"
        (sem / "training/images").mkdir(parents=True)
        (sem / "training/v2.0/labels").mkdir(parents=True)
        (sem / "config_v2.0.json").write_text(json.dumps({"labels": [
            {"readable": "rail"}, {"readable": "ground"}]}))
        Image.fromarray(rgb).save(sem / "training/images/rs08006.jpg")
        labels = np.zeros((48, 96), dtype=np.uint8)
        labels[:, 32:] = 1
        Image.fromarray(labels).save(sem / "training/v2.0/labels/rs08006.png")

    def reason(self, mode="structured", explanatory=0.0, flip=0.0):
        return ReasonSegDataset(str(self.root), None, "unused", image_size=1024,
            num_classes_per_sample=2, reason_seg_data="ReasonSegRail|train",
            explanatory=explanatory, weight_map_weight=4.0,
            counterfactual_flip_prob=flip, sample_mode=mode)

    def draw(self, dataset, choice=None):
        random.seed(31)
        np.random.seed(31)
        if choice is None:
            return dataset[0]
        with patch("utils.reason_seg_dataset.random.randint", side_effect=[0, choice]):
            return dataset[0]

    def test_structured_path_never_loads_clip_or_llava(self):
        with patch("transformers.CLIPImageProcessor.from_pretrained",
                   side_effect=AssertionError("CLIP must not load")):
            self.draw(self.reason(), 1).validate()
            SemSegDataset(str(self.root), None, "unused", sem_seg_data="railsem",
                          sample_mode="structured")[0].validate()
            ValDataset(str(self.root), None, "unused", "ReasonSeg|val",
                       sample_mode="structured")[0].validate()
        self.assertFalse(any(name.startswith("model.llava") for name in sys.modules))

    def test_reason_mask_text_and_explanation_modes(self):
        mask = self.draw(self.reason(), 0)
        explanation = self.draw(self.reason(), 1)
        text = self.draw(self.reason(explanatory=1.0))
        self.assertEqual(mask.answers, [MASK_ANSWER] * 2)
        self.assertEqual(explanation.answers, [MASK_ANSWER + " " + EXPLANATION] * 2)
        self.assertEqual(text.answers, [EXPLANATION] * 2)
        self.assertEqual(explanation.mask_offsets, [0, 1, 2])
        self.assertEqual(text.mask_offsets, [0, 0, 0])
        self.assertEqual(tuple(text.masks.shape), (0, 48, 96))
        self.assertEqual(text.reason_seg_weight_maps.numel(), 0)

    def test_flip_preserves_image_mask_weights_and_ego_alignment(self):
        original = self.draw(self.reason(), 1)
        flipped = self.draw(self.reason(flip=1.0), 1)
        np.testing.assert_array_equal(flipped.rgb_image, original.rgb_image[:, ::-1])
        torch.testing.assert_close(flipped.masks, original.masks.flip(-1))
        torch.testing.assert_close(flipped.reason_seg_weight_maps,
                                   original.reason_seg_weight_maps.flip(-1))
        self.assertEqual((original.rail_ego_side_label, flipped.rail_ego_side_label), (1, 0))
        self.assertIn("The right blade is closed and the left blade is open.", flipped.answers[0])
        self.assertIn("follows the left-hand path.", flipped.answers[0])
        self.assertTrue(flipped.metadata["flipped"])
        resized = self.reason().transform.apply_image(flipped.rgb_image)
        expected_sam = self.reason().preprocess(torch.from_numpy(resized).permute(2, 0, 1))
        torch.testing.assert_close(flipped.sam_image, expected_sam)

    def legacy_modules(self):
        # Only the legacy conversation formatter, not its incompatible model imports.
        spec = importlib.util.spec_from_file_location("_test_legacy_conversation",
            ROOT / "model/llava/conversation.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(module)
        return {"model.llava": SimpleNamespace(conversation=module)}

    def test_legacy_and_structured_reason_samples_remain_identical(self):
        processor = SimpleNamespace(preprocess=lambda image, **kw:
            {"pixel_values": torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)})
        for choice in (0, 1):
            with self.subTest(choice=choice):
                structured = self.draw(self.reason(flip=1.0), choice)
                with patch("transformers.CLIPImageProcessor.from_pretrained", return_value=processor), \
                     patch.dict(sys.modules, self.legacy_modules()):
                    legacy = self.draw(self.reason(mode="llava", flip=1.0), choice)
                self.assertEqual(len(legacy), 11)
                for index, field in ((1, "sam_image"), (4, "masks"), (5, "label"),
                                     (9, "reason_seg_weight_maps")):
                    torch.testing.assert_close(legacy[index], getattr(structured, field))
                np.testing.assert_array_equal(legacy[2].permute(1, 2, 0).numpy(), structured.rgb_image)
                self.assertEqual(legacy[7], structured.questions)
                self.assertEqual(legacy[8], structured.sampled_classes)
                self.assertEqual(legacy[10], structured.rail_ego_side_label)
                for conversation, answer in zip(legacy[3], structured.answers):
                    self.assertIn(answer, conversation)

    def test_semantic_masks_follow_sampled_classes(self):
        sample = SemSegDataset(str(self.root), None, "unused", sem_seg_data="railsem",
                               num_classes_per_sample=2, sample_mode="structured")[0]
        for name, mask in zip(sample.sampled_classes, sample.masks):
            torch.testing.assert_close(mask, sample.label.eq(["rail", "ground"].index(name)))
        self.assertEqual(sample.mask_offsets, [0, 1, 2])

    def test_validation_remains_reasonseg_val_and_prompt_only(self):
        sample = ValDataset(str(self.root), None, "unused", "ReasonSeg|val",
                            sample_mode="structured")[0]
        self.assertIn("/ReasonSeg/val/", sample.image_path)
        self.assertTrue(sample.inference)
        self.assertEqual(sample.answers, ["[SEG]."])

    def test_hybrid_structured_mode_preserves_default_sampling_contract(self):
        dataset = HybridDataset(str(self.root), None, "unused", dataset="sem_seg||reason_seg_rail",
            sample_rate=[1, 1], sem_seg_data="railsem", reason_seg_rail_data="ReasonSegRail|train",
            explanatory=-1, sample_mode="structured")
        self.assertFalse(dataset[0].inference)
        with self.assertRaisesRegex(ValueError, "Structured"):
            HybridDataset(str(self.root), None, "unused", dataset="vqa", sample_mode="structured")


@unittest.skipIf(ML_ERROR, "Stage 2 ML dependencies unavailable: " + str(ML_ERROR))
class NativeProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_dir = os.environ.get("QWEN_TEST_MODEL_DIR")
        if not model_dir:
            raise unittest.SkipTest("Set QWEN_TEST_MODEL_DIR for real offline Qwen processor tests")
        cls.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        cls.collator = QwenSegCollator(cls.processor, use_rail_reasoning_weighted_ce=True,
                                       use_seg_token_weighted_ce=True)

    def sample(self, size=(256, 256), answers=None, inference=False, rail=False):
        answers = answers or [MASK_ANSWER]
        h, w = size
        mask_count = sum(answer.count("[SEG]") for answer in answers)
        return StructuredSegSample("fixture.jpg", np.zeros((h, w, 3), dtype=np.uint8),
            torch.zeros(3, 1024, 1024), ["<image>\nExplain this route."] * len(answers),
            answers, torch.zeros(mask_count, h, w), torch.full((h, w), 255),
            (h, w), ["route"], torch.empty(0), rail_ego_side_label=1,
            is_rail_reasoning=rail, inference=inference)

    def test_square_wide_narrow_visual_counts_and_assistant_boundaries(self):
        for size in ((256, 256), (256, 1024), (1024, 256)):
            with self.subTest(size=size):
                batch = self.collator([self.sample(size)])
                ids, labels = batch["input_ids"][0], batch["labels"][0]
                self.assertEqual(decode_exact(self.processor.tokenizer, labels[labels.ne(-100)].tolist()),
                                 MASK_ANSWER + "<|im_end|>")
                self.assertTrue(labels[ids.eq(self.processor.image_token_id)].eq(-100).all())
                grid = batch["image_grid_thw"][0]
                self.assertEqual(batch["visual_token_counts"][0],
                                 int(grid.prod()) // self.processor.image_processor.merge_size**2)
                self.assertEqual(int(ids.eq(self.processor.image_token_id).sum()),
                                 batch["visual_token_counts"][0])
                span = batch["assistant_spans"][0]
                self.assertTrue(labels[:span["token_start"]].eq(-100).all())
                self.assertTrue(labels[span["token_end"]:].eq(-100).all())
                self.assertEqual(tuple(batch["images"].shape), (1, 3, 1024, 1024))

    def test_contextual_rail_fields_and_existing_weights(self):
        answer = MASK_ANSWER + " " + EXPLANATION
        batch = self.collator([self.sample(answers=[answer, EXPLANATION], rail=True)])
        expected = RAIL_REASONING_DECISION_PATTERN.search(answer).groupdict()
        for row, groups in enumerate(batch["reasoning_group_positions"]):
            for name, positions in groups.items():
                decoded = decode_exact(self.processor.tokenizer, batch["input_ids"][row, positions].tolist())
                self.assertEqual(decoded.strip(), expected[name])
                self.assertTrue(batch["labels"][row, positions].ne(-100).all())
                self.assertTrue(batch["ce_token_weights"][row, positions].eq(
                    60.0 if name in ("switch", "right_state") else 1.0).all())
        self.assertEqual(batch["mask_offsets_list"], [[0, 1, 1]])
        self.assertEqual(len(batch["seg_token_positions"][1]), 0)
        positions = batch["seg_token_positions"][0]
        self.assertTrue(batch["ce_token_weights"][0, positions].eq(3.0).all())

    def test_multiple_images_conversations_masks_and_padding(self):
        first = self.sample(answers=[MASK_ANSWER, "Only text.", "[SEG] and [SEG]."])
        second = self.sample((256, 512), answers=["[SEG]."])
        batch = self.collator([first, second])
        self.assertEqual(batch["offset"].tolist(), [0, 3, 4])
        self.assertEqual(batch["conversation_to_image"].tolist(), [0, 0, 0, 1])
        self.assertEqual(batch["mask_offsets_list"], [[0, 1, 1, 3], [0, 1]])
        self.assertEqual(batch["image_grid_thw"].shape[0], 4)
        self.assertEqual(batch["images"].shape[0], 2)
        self.assertTrue(batch["labels"][batch["attention_mask"].eq(0)].eq(-100).all())
        self.assertTrue(batch["rail_ego_side_labels"].eq(-100).all())

    def test_prompt_only_inference_has_no_answer_leak(self):
        batch = self.collator([self.sample(answers=["UNIQUE_ANSWER [SEG]."], inference=True)])
        self.assertTrue(batch["labels"].eq(-100).all())
        self.assertNotIn("UNIQUE_ANSWER", decode_exact(self.processor.tokenizer, batch["input_ids"][0].tolist()))
        self.assertEqual(batch["seg_token_positions"], [[]])

    def test_generation_uses_left_padding_for_different_prompt_lengths(self):
        first = self.sample(inference=True)
        second = replace(self.sample(inference=True),
                         questions=["<image>\nA considerably longer question about the right route."])
        batch = self.collator([first, second])
        self.assertEqual(batch["attention_mask"][:, -1].tolist(), [1, 1])
        self.assertEqual(int(batch["attention_mask"][0, 0]), 0)
        self.assertTrue(batch["labels"].eq(-100).all())
        expected = self.collator([first])["input_ids"][0]
        torch.testing.assert_close(batch["input_ids"][0, batch["attention_mask"][0].bool()], expected)

    def test_atomic_seg_and_processor_config_round_trip(self):
        self.assertEqual(self.processor.tokenizer.decode([self.collator.seg_token_id],
                                                        skip_special_tokens=True), "[SEG]")
        with tempfile.TemporaryDirectory() as temp:
            self.processor.save_pretrained(temp)
            self.collator.config.save_pretrained(temp)
            restored = AutoProcessor.from_pretrained(temp, local_files_only=True)
            config = QwenProcessingConfig.from_pretrained(temp)
            collator = QwenSegCollator(restored, config)
            self.assertEqual(collator.seg_token_id, self.collator.seg_token_id)
            original = self.collator([self.sample()])
            reloaded = collator([self.sample()])
            for key in ("input_ids", "labels", "pixel_values", "image_grid_thw"):
                torch.testing.assert_close(original[key], reloaded[key])

    def test_overlength_and_mixed_modes_fail_instead_of_truncating(self):
        small = QwenSegCollator(self.processor, QwenProcessingConfig(max_seq_length=4))
        with self.assertRaisesRegex(ValueError, "no truncation"):
            small([self.sample()])
        with self.assertRaisesRegex(ValueError, "Cannot mix"):
            self.collator([self.sample(), self.sample(inference=True)])

    def test_nonrail_unicode_answer_round_trip(self):
        batch = self.collator([self.sample(answers=["The route is left. café 🚂 [SEG]."])])
        labels = batch["labels"][0]
        self.assertEqual(decode_exact(self.processor.tokenizer, labels[labels.ne(-100)].tolist()),
                         "The route is left. café 🚂 [SEG].<|im_end|>")


if __name__ == "__main__":
    unittest.main()
