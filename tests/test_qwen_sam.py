"""Small native Qwen + real SAM prompt/decoder tests, CPU and no downloaded weights.

The reduced fixtures test plumbing/gradients, not 8B/ViT-H hardware feasibility.
Missing runtime dependencies skip locally and must not count as a verified gate.
"""

from copy import deepcopy
from dataclasses import replace
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ML_ERROR = None
try:
    import torch
    from torch import nn
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from model.QwenSAM import QwenSAM, read_sam_checkpoint, seg_event_positions, validate_sam_shapes
    from model.qwen_sam_config import QwenSAMConfig
    from model.qwen_seg_tokens import SegTokenEmbedding, SegTokenOutput, install_seg_token_rows, materialized_seg_rows
    from model.segment_anything.modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
except ModuleNotFoundError as exc:
    ML_ERROR = str(exc)


class FixtureTokenizer:
    def add_tokens(self, *args, **kwargs):
        return 0

    def __len__(self):
        return 61

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[60])

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return "".join("[SEG]" if token == 60 else "" if token in (0, 51) and skip_special_tokens
                       else f"token{token} " for token in ids)


def tiny_qwen():
    config = Qwen3VLConfig(
        text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "max_position_embeddings": 128, "tie_word_embeddings": False,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
            "bos_token_id": 50, "eos_token_id": 51, "pad_token_id": 0},
        vision_config={"depth": 4, "hidden_size": 32, "intermediate_size": 64,
            "num_heads": 4, "patch_size": 2, "temporal_patch_size": 2,
            "spatial_merge_size": 2, "out_hidden_size": 32,
            "num_position_embeddings": 64, "deepstack_visual_indexes": [0, 1]},
        image_token_id=20, video_token_id=23, vision_start_token_id=21, vision_end_token_id=22,
        tie_word_embeddings=False, bos_token_id=50, eos_token_id=51, pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    return Qwen3VLForConditionalGeneration(config)


def tiny_sam():
    class ImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_size = 8
            self.patch_embed = nn.Conv2d(3, 256, 4, stride=4)
            self.blocks = nn.ModuleList([nn.Conv2d(256, 256, 1)])
            self.neck = nn.Conv2d(256, 256, 1)

        def forward(self, images):
            return self.neck(self.blocks[0](self.patch_embed(images)))

    return Sam(ImageEncoder(), PromptEncoder(embed_dim=256, image_embedding_size=(2, 2),
        input_image_size=(8, 8), mask_in_chans=16), MaskDecoder(transformer_dim=256,
        transformer=TwoWayTransformer(depth=1, embedding_dim=256, num_heads=8, mlp_dim=512)))


@unittest.skipIf(ML_ERROR, "Stage 3 ML dependencies unavailable: " + str(ML_ERROR))
class SegRowsTest(unittest.TestCase):
    def test_rows_update_but_native_vocabulary_matrices_stay_frozen(self):
        embedding = SegTokenEmbedding(nn.Embedding(64, 8), 60)
        output = SegTokenOutput(nn.Linear(8, 64, bias=False), 60)
        input_base, output_base = embedding.weight.clone(), output.weight.clone()
        optimizer = torch.optim.SGD([embedding.seg_row, output.seg_row], lr=0.1)
        loss = nn.functional.cross_entropy(output(embedding(torch.tensor([[1, 60]]))).view(-1, 64),
                                            torch.tensor([60, 2]))
        loss.backward()
        self.assertIsNotNone(embedding.seg_row.grad)
        self.assertIsNotNone(output.seg_row.grad)
        self.assertIsNone(embedding.weight.grad)
        self.assertIsNone(output.weight.grad)
        input_row, output_row = embedding.seg_row.clone(), output.seg_row.clone()
        optimizer.step()
        self.assertFalse(torch.equal(input_row, embedding.seg_row))
        self.assertFalse(torch.equal(output_row, output.seg_row))
        torch.testing.assert_close(embedding.weight, input_base)
        torch.testing.assert_close(output.weight, output_base)

    def test_no_seg_still_gives_an_allocated_zero_input_row_gradient(self):
        embedding = SegTokenEmbedding(nn.Embedding(64, 8), 60)
        embedding(torch.tensor([[1, 2]])).sum().backward()
        torch.testing.assert_close(embedding.seg_row.grad, torch.zeros(8))

    def test_native_vocab_not_shrunk_and_export_rows_restore_after_exception(self):
        qwen = tiny_qwen()
        with patch.object(qwen, "resize_token_embeddings", wraps=qwen.resize_token_embeddings) as resize:
            self.assertEqual(install_seg_token_rows(qwen, FixtureTokenizer(), 60), 64)
            resize.assert_not_called()
        embedding, output = qwen.get_input_embeddings(), qwen.get_output_embeddings()
        original_input, original_output = embedding.weight.clone(), output.weight.clone()
        with torch.no_grad():
            embedding.seg_row.fill_(0.25)
            output.seg_row.fill_(0.5)
        with self.assertRaisesRegex(RuntimeError, "save failure"):
            with materialized_seg_rows(qwen):
                self.assertIsInstance(qwen.get_input_embeddings(), nn.Embedding)
                torch.testing.assert_close(qwen.get_input_embeddings().weight[60], embedding.seg_row)
                torch.testing.assert_close(qwen.get_output_embeddings().weight[60], output.seg_row)
                raise RuntimeError("save failure")
        self.assertIs(qwen.get_input_embeddings(), embedding)
        self.assertIs(qwen.get_output_embeddings(), output)
        torch.testing.assert_close(embedding.weight, original_input)
        torch.testing.assert_close(output.weight, original_output)

    def test_saved_native_qwen_loads_without_custom_token_wrapper_classes(self):
        qwen = tiny_qwen()
        install_seg_token_rows(qwen, FixtureTokenizer(), 60)
        with torch.no_grad():
            qwen.get_input_embeddings().seg_row.fill_(0.25)
            qwen.get_output_embeddings().seg_row.fill_(0.5)
        with tempfile.TemporaryDirectory() as temp:
            with materialized_seg_rows(qwen):
                qwen.save_pretrained(temp, safe_serialization=True)
            restored = Qwen3VLForConditionalGeneration.from_pretrained(temp, local_files_only=True)
            self.assertEqual(restored.get_input_embeddings().weight.shape[0], 64)
            torch.testing.assert_close(restored.get_input_embeddings().weight[60], torch.full((32,), 0.25))
            torch.testing.assert_close(restored.get_output_embeddings().weight[60], torch.full((32,), 0.5))

    def test_vocab_grows_only_when_added_token_exceeds_native_capacity(self):
        class ExtendedTokenizer(FixtureTokenizer):
            def __len__(self):
                return 80

        qwen = tiny_qwen()
        self.assertEqual(install_seg_token_rows(qwen, ExtendedTokenizer(), 79), 80)
        self.assertEqual(tuple(qwen.get_input_embeddings().weight.shape), (80, 32))
        self.assertEqual(tuple(qwen.get_output_embeddings().weight.shape), (80, 32))
        self.assertTrue(torch.isfinite(qwen.get_input_embeddings().seg_row).all())

    def test_install_rejects_tied_vocab_before_replacing_modules(self):
        qwen = tiny_qwen()
        embedding = qwen.get_input_embeddings()
        qwen.get_output_embeddings().weight = embedding.weight
        with self.assertRaisesRegex(ValueError, "untied"):
            install_seg_token_rows(qwen, FixtureTokenizer(), 60)
        self.assertIs(qwen.get_input_embeddings(), embedding)


@unittest.skipIf(ML_ERROR, "Stage 3 ML dependencies unavailable: " + str(ML_ERROR))
class CompositeModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)

    def model(self, bridge="single"):
        return QwenSAM(tiny_qwen(), tiny_sam(), SimpleNamespace(tokenizer=FixtureTokenizer()),
                       QwenSAMConfig(seg_prompt_bridge_type=bridge))

    def batch(self, counts=(1, 0, 2)):
        rows = []
        labels = []
        for count in counts:
            # Four native visual tokens; a prompt SEG must never trigger a mask.
            prompt = [50, 21, 20, 20, 20, 20, 22, 60, 7]
            answer = [60] * count + [8, 51]
            rows.append(torch.tensor(prompt + answer))
            labels.append(torch.tensor([-100] * len(prompt) + answer))
        ids = nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0)
        targets = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": ids, "attention_mask": ids.ne(0).long(), "labels": targets,
            "pixel_values": torch.randn(len(counts) * 16, 24),
            "image_grid_thw": torch.tensor([[1, 4, 4]] * len(counts)),
            "images": torch.randn(2, 3, 8, 8), "offset": torch.tensor([0, 2, len(counts)]),
            "resize_list": [(8, 8)] * 2, "original_size_list": [(4, 6)] * 2,
            "masks_list": [torch.zeros(sum(counts[:2]), 4, 6), torch.zeros(sum(counts[2:]), 4, 6)]}

    def test_real_sam_decodes_one_mask_per_seg_with_both_bridges(self):
        for bridge, queries in (("single", 1), ("four_query", 4)):
            with self.subTest(bridge=bridge):
                model = self.model(bridge).eval()
                result = model(**self.batch())
                self.assertEqual(result["seg_token_counts"].tolist(), [1, 0, 2])
                self.assertEqual([tuple(mask.shape) for mask in result["pred_masks"]], [(1, 4, 6), (2, 4, 6)])
                self.assertEqual([tuple(group.shape) for group in result["prompt_groups"]],
                                 [(1, queries, 256), (2, queries, 256)])
                self.assertEqual(result["seg_events"].tolist(), [[0, 9], [2, 9], [2, 10]])
                self.assertEqual(result["seg_predictor_positions"].tolist(), [8, 8, 9])
                self.assertTrue(torch.isfinite(result["logits"]).all())
                self.assertTrue(all(torch.isfinite(mask).all() for mask in result["pred_masks"]))

    def test_zero_seg_has_empty_masks_and_all_auxiliary_gradients(self):
        for bridge in ("single", "four_query"):
            with self.subTest(bridge=bridge):
                model = self.model(bridge)
                result = model(**self.batch((0, 0, 0)))
                self.assertEqual([tuple(mask.shape) for mask in result["pred_masks"]], [(0, 4, 6)] * 2)
                result["keepalive_loss"].backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and not name.startswith("qwen."):
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_mask_and_ce_backward_reach_fresh_bridge_rows_and_selected_sam(self):
        model = self.model("four_query")
        batch = self.batch()
        result = model(**batch)
        logits = result["logits"][:, :-1].reshape(-1, 64)
        loss = nn.functional.cross_entropy(logits, batch["labels"][:, 1:].reshape(-1))
        loss = loss + sum(mask.square().mean() for mask in result["pred_masks"]) + result["keepalive_loss"]
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertIsNone(model.qwen.get_input_embeddings().weight.grad)
        self.assertIsNone(model.qwen.get_output_embeddings().weight.grad)

    def test_four_query_cannot_access_padding_or_future_tokens(self):
        model = self.model("four_query").eval()
        states = tuple(torch.randn(1, 8, 32) for _ in range(5))
        ids = torch.tensor([[0, 7, 8, 60, 9, 10, 11, 0]])
        attention = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]])
        assistant = ids.eq(60)
        first = model._prompts(states, ids, attention, assistant)[0]
        changed = tuple(state.clone() for state in states)
        for state in changed:
            state[:, 0] = float("nan")
            state[:, 4:] = 9999
        second = model._prompts(changed, ids, attention, assistant)[0]
        torch.testing.assert_close(first, second)
        altered_anchor = tuple(state.clone() for state in states)
        altered_anchor[-1][:, 3] += 5
        self.assertFalse(torch.equal(first, model._prompts(altered_anchor, ids, attention, assistant)[0]))

    def test_native_final_state_is_normalized_and_layer_count_configuration_driven(self):
        model = self.model("four_query").eval()
        batch = self.batch()
        _, states = model._language_forward(**{key: batch[key] for key in
            ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")})
        self.assertEqual(len(states), 5)
        reference = model.qwen.model(**{key: batch[key] for key in
            ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}, use_cache=False,
            output_hidden_states=False, return_dict=True).last_hidden_state
        torch.testing.assert_close(states[-1], reference)

    def test_nonreentrant_checkpointing_preserves_vision_and_bridge_gradients(self):
        model = self.model("four_query")
        model.qwen.visual.blocks[-1].requires_grad_(True)
        model.qwen.visual.merger.requires_grad_(True)
        model.gradient_checkpointing_enable()
        result = model(**self.batch())
        loss = sum(mask.square().mean() for mask in result["pred_masks"]) + result["keepalive_loss"]
        loss.backward()
        for name, parameter in model.qwen.visual.blocks[-1].named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(model.seg_query_bridge.memory_projection.weight.grad.abs().sum()), 0)

    def test_left_padding_and_prompt_seg_are_not_mask_events(self):
        ids = torch.tensor([[0, 0, 60, 7, 60, 60]])
        valid = torch.tensor([[0, 0, 1, 1, 1, 1]])
        assistant = torch.tensor([[0, 0, 0, 0, 1, 1]])
        _, events, predictors = seg_event_positions(ids, valid, assistant, 60)
        self.assertEqual(events.tolist(), [[0, 4], [0, 5]])
        self.assertEqual(predictors.tolist(), [3, 4])

    def test_generation_fresh_pass_preserves_rows_and_reports_missing_seg(self):
        model = self.model("four_query")
        batch = self.batch((0, 0, 0))
        batch["input_ids"] = batch["input_ids"][:, :9]
        batch["attention_mask"] = batch["attention_mask"][:, :9]
        batch["inference"] = True
        suffix = torch.tensor([[60, 51, 0, 0], [8, 51, 0, 0], [60, 60, 8, 51]])
        generated = torch.cat([batch["input_ids"], suffix], dim=1)
        with patch.object(model.qwen, "generate", return_value=generated), \
             patch.object(model, "_language_forward", wraps=model._language_forward) as full_pass:
            result = model.generate_masks(batch, max_new_tokens=4)
            full_pass.assert_called_once()
        self.assertTrue(model.training)
        self.assertEqual(result["seg_token_counts"].tolist(), [1, 0, 2])
        self.assertEqual(result["missing_seg_conversations"], [1])
        self.assertEqual(result["attention_mask"][0, -2:].tolist(), [0, 0])
        self.assertEqual(result["answers"][0], "[SEG]")

    def test_native_cached_generation_works_with_seg_vocabulary_wrappers(self):
        model = self.model().eval()
        batch = self.batch((0, 0, 0))
        batch["input_ids"] = batch["input_ids"][:, :9]
        batch["attention_mask"] = batch["attention_mask"][:, :9]
        batch["inference"] = True
        result = model.generate_masks(batch, max_new_tokens=2, do_sample=False,
                                      suppress_tokens=[0, 20, 21, 22, 23])
        self.assertEqual(result["output_ids"].shape[0], 3)
        self.assertEqual(len(result["answers"]), 3)
        self.assertLessEqual(result["output_ids"].shape[1], 11)

    def test_composite_aux_restore_is_exact_and_partial_resume_rejected(self):
        model = self.model("four_query")
        state = deepcopy(model.auxiliary_state_dict())
        with torch.no_grad():
            model.seg_query_bridge.learned_queries.add_(1)
        model.load_auxiliary_state_dict(state)
        for name, tensor in model.seg_query_bridge.state_dict().items():
            torch.testing.assert_close(tensor, state["seg_query_bridge"][name])
        broken = deepcopy(state)
        broken["seg_query_bridge"].pop("residual_gate")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            model.load_auxiliary_state_dict(broken)

    def test_full_composite_save_reload_preserves_masks_and_seg_rows(self):
        model = self.model("four_query").eval()
        batch = self.batch()
        with torch.no_grad():
            model.qwen.get_input_embeddings().seg_row.add_(0.1)
            model.qwen.get_output_embeddings().seg_row.add_(0.2)
            original = model(**batch)
        processor = SimpleNamespace(tokenizer=FixtureTokenizer(), save_pretrained=lambda directory: None)
        model.processor = processor
        module = importlib.import_module("model.QwenSAM")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "composite"
            model.save_pretrained(directory)
            with patch.object(module.AutoProcessor, "from_pretrained", return_value=processor), \
                 patch.object(module, "build_sam_vit_h", side_effect=lambda **kwargs: tiny_sam()):
                restored = QwenSAM.from_pretrained(directory, dtype=torch.float32).eval()
            with torch.no_grad():
                reloaded = restored(**batch)
            torch.testing.assert_close(original["logits"], reloaded["logits"])
            for first, second in zip(original["pred_masks"], reloaded["pred_masks"]):
                torch.testing.assert_close(first, second)
            self.assertEqual(restored.config, model.config)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                model.save_pretrained(directory)

    def test_sam_selection_loads_only_complete_matching_shapes(self):
        sam = tiny_sam()
        rail = {"model.visual_model." + key: value for key, value in sam.state_dict().items()}
        rail["model.embed_tokens.weight"] = torch.full((2, 2), float("nan"))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rail.pt"
            torch.save(rail, path)
            state = read_sam_checkpoint(path, sam.state_dict().keys(), "rail_checkpoint")
        validate_sam_shapes(sam, state)
        self.assertEqual(set(state), set(sam.state_dict()))
        broken = dict(state)
        first = next(iter(broken))
        broken[first] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_sam_shapes(sam, broken)

    def test_sharded_hf_rail_checkpoint_is_read_without_language_tensors(self):
        from safetensors.torch import save_file
        import json

        sam = tiny_sam()
        tensors = {"model.visual_model." + key: value.contiguous() for key, value in sam.state_dict().items()}
        keys = list(tensors)
        split = len(keys) // 2
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            first = {key: tensors[key] for key in keys[:split]}
            second = {key: tensors[key] for key in keys[split:]}
            save_file(first, directory / "sam-1.safetensors")
            save_file(second, directory / "sam-2.safetensors")
            index = {key: "sam-1.safetensors" if key in first else "sam-2.safetensors" for key in keys}
            # This unrelated language shard intentionally does not exist.
            index["model.embed_tokens.weight"] = "language-not-loaded.safetensors"
            (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
            restored = read_sam_checkpoint(directory, sam.state_dict().keys(), "rail_checkpoint")
        for key, tensor in sam.state_dict().items():
            torch.testing.assert_close(restored[key], tensor)

    def test_fresh_load_requires_explicit_sam_choice(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            QwenSAM.from_native_pretrained("unused", sam_init_source=None, sam_init_path=None)


if __name__ == "__main__":
    unittest.main()
