"""Lightweight Stage 2 checks; native processor coverage lives in the ML suite."""

from dataclasses import replace
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

from utils.qwen_token_alignment import (
    ASSISTANT_END, assistant_positions, expand_image_for_alignment,
    question_content, reasoning_positions, render_qwen_pair,
)
from utils.segmentation_sample import StructuredSegSample, validate_sample_mode


class CharacterTokenizer:
    def decode(self, ids, **kwargs):
        return "".join(chr(value) for value in ids)


class AlignmentTest(unittest.TestCase):
    def test_standalone_image_marker_only_is_replaced(self):
        self.assertEqual(question_content("<image>\nDescribe the left rail."), [
            {"type": "image"}, {"type": "text", "text": "Describe the left rail."},
        ])
        self.assertEqual(question_content("Before\n<image>\nAfter"), [
            {"type": "text", "text": "Before\n"}, {"type": "image"},
            {"type": "text", "text": "After"},
        ])
        self.assertEqual(question_content("Explain <image> literally.")[1]["text"],
                         "Explain <image> literally.")

    def test_duplicate_images_and_control_tokens_rejected(self):
        for question in ("<image>\n<image>\nExplain", "<|im_end|>prompt"):
            with self.subTest(question=question), self.assertRaises(ValueError):
                question_content(question)

    def test_native_template_answer_is_not_rewritten(self):
        processor = SimpleNamespace(apply_chat_template=lambda messages, **kw:
            "prefix" if kw["add_generation_prompt"] else
            "prefix" + messages[-1]["content"] + ASSISTANT_END + "\n")
        answer = "The segmentation result is [SEG]. café\n"
        text, start = render_qwen_pair(processor, "<image>\nQuestion?", answer)
        self.assertEqual(text[start:], answer + ASSISTANT_END + "\n")
        self.assertEqual(render_qwen_pair(processor, "Question?", answer, True), ("prefix", None))
        processor.apply_chat_template = lambda *a, **kw: "changed"
        with self.assertRaisesRegex(ValueError, "template changed"):
            render_qwen_pair(processor, "Question?", answer)

    def test_answer_control_tokens_rejected(self):
        with self.assertRaisesRegex(ValueError, "control token"):
            render_qwen_pair(None, "Question", "<|image_pad|>")

    def test_native_visual_expansion_checked(self):
        self.assertEqual(expand_image_for_alignment("x<image_pad>y", "<image_pad>", 3),
                         "x<image_pad><image_pad><image_pad>y")
        for text, count in (("no image", 1), ("<i><i>", 1), ("<i>", 0)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                expand_image_for_alignment(text, "<i>", count)

    def test_assistant_offsets_ignore_prefix_and_preserve_unicode(self):
        text = "prompt: café [SEG]." + ASSISTANT_END + "\n"
        positions = assistant_positions(list(map(ord, text)),
            [(i, i + 1) for i in range(len(text))], text, 8, len(text) - 1,
            CharacterTokenizer())
        self.assertEqual(positions, list(range(8, len(text) - 1)))

    def test_cross_boundary_token_and_broken_round_trip_rejected(self):
        with self.assertRaisesRegex(ValueError, "crosses"):
            assistant_positions([1], [(0, 3)], "abc", 1, 3, CharacterTokenizer())
        with self.assertRaisesRegex(ValueError, "round-trip"):
            assistant_positions([ord("x")], [(0, 1)], "a", 0, 1, CharacterTokenizer())

    def test_reasoning_uses_actual_context_offsets(self):
        text = "prompt\n  This is a turnout switch."
        answer = text[7:]
        pattern = re.compile(r"This is a (?P<switch>turnout|merge) switch\.$")
        groups = reasoning_positions(answer, [(i, i + 1) for i in range(len(text))],
            7, list(map(ord, text)), CharacterTokenizer(), pattern, {"[SEG]."})
        self.assertEqual("".join(text[i] for i in groups["switch"]), "turnout")
        self.assertEqual(reasoning_positions("[SEG].", [], 0, [], CharacterTokenizer(),
                                            pattern, {"[SEG]."}), {})
        with self.assertRaisesRegex(ValueError, "canonical"):
            reasoning_positions("unstructured", [], 0, [], CharacterTokenizer(), pattern, set())


def array(shape, dtype="float32"):
    return SimpleNamespace(shape=shape, dtype=dtype, numel=lambda: 0)


class SampleContractTest(unittest.TestCase):
    def setUp(self):
        self.sample = StructuredSegSample("image.jpg", array((8, 16, 3), "uint8"),
            array((3, 1024, 1024)), ["q1", "q2", "q3"],
            ["[SEG].", "text only", "[SEG]."], array((2, 8, 16)), array((8, 16)),
            (8, 16), ["one", "two"], array((0,)))

    def test_zero_one_multiple_mask_ordering(self):
        self.assertIs(self.sample.validate(), self.sample)
        self.assertEqual(self.sample.mask_offsets, [0, 1, 1, 2])
        self.assertEqual(self.sample.original_size, (8, 16))
        replace(self.sample, answers=["text"] * 3, masks=array((0, 8, 16))).validate()
        replace(self.sample, answers=["[SEG] [SEG]", "text", "text"]).validate()

    def test_inconsistent_structured_data_rejected(self):
        cases = ({"questions": []}, {"answers": ["[SEG]."]},
                 {"rgb_image": array((8, 16, 3), "float32")},
                 {"masks": array((3, 8, 16))}, {"masks": array((2, 16, 8))},
                 {"label": array((16, 8))}, {"sam_image": array((8, 16, 3))},
                 {"reason_seg_weight_maps": SimpleNamespace(shape=(1, 8, 16), numel=lambda: 128)})
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                replace(self.sample, **changed).validate()

    def test_only_explicit_backend_modes(self):
        self.assertEqual(validate_sample_mode("llava"), "llava")
        self.assertEqual(validate_sample_mode("structured"), "structured")
        with self.assertRaises(ValueError):
            validate_sample_mode("qwen")

    def test_legacy_validation_default_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        self.assertIn('default="ReasonSeg|val"', (root / "train_ds.py").read_text())


if __name__ == "__main__":
    unittest.main()
