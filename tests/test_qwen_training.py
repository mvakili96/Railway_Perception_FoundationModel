"""CPU tests for actual PEFT training, losses, full-state restore and merge."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

ML_ERROR = None
try:
    import torch
    import peft
    import test_qwen_sam as fixtures
    from utils.qwen_training import TrainingPolicy, configure_training, compute_losses, deepspeed_config, move_batch
    from utils.qwen_checkpoint import check_resume_settings, resolve_checkpoint_tag
    from utils.qwen_checkpoint import save_run_metadata, reconstruct_training_model
    from utils.qwen_checkpoint import assert_complete_state, prune_completed_checkpoints
    from utils.qwen_evaluation import foreground_iou, summarize_test_results
    from model.segmentation_losses import dice_loss, sigmoid_ce_loss
    from model.qwen_seg_tokens import materialized_seg_rows
    from transformers import Qwen3VLForConditionalGeneration
    from train_qwen_sam import parse_args
    from chat_batch_qwen import load_requests
except ModuleNotFoundError as exc:
    ML_ERROR = str(exc)


@unittest.skipIf(ML_ERROR, "Stage 4 dependencies unavailable: " + str(ML_ERROR))
class QwenTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.fixture = fixtures.CompositeModelTest()

    def batch(self, counts=(1, 0, 2)):
        batch = self.fixture.batch(counts)
        batch.update(ce_token_weights=torch.ones_like(batch["labels"], dtype=torch.float32),
            rail_ego_side_labels=torch.tensor([1, -100, 0]),
            reason_seg_weight_maps_list=[torch.empty(0), torch.empty(0)],
            rail_switch_token_masks=batch["labels"].eq(8),
            rail_right_state_token_masks=batch["labels"].eq(8))
        return batch

    def test_trainability_after_peft_and_exact_groups(self):
        model = self.fixture.model()
        groups, manifest = configure_training(model, TrainingPolicy(vision_last_blocks=2))
        self.assertEqual([g["group_name"] for g in groups],
                         ["language_lora", "vision", "bridge_and_seg", "sam"])
        self.assertEqual([g["lr"] for g in groups], [1e-4, 1e-5, 1e-4, 1e-4])
        names = [n for group in manifest for n in group["parameter_names"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(any(n.endswith("embed_tokens.seg_row") for n in names))
        self.assertTrue(any(n.endswith("lm_head.seg_row") for n in names))
        self.assertFalse(any("visual" in n and "lora" in n for n in names))
        self.assertTrue(any("deepstack_merger_list" in n for n in names))
        visual = model.native_qwen.model.visual
        self.assertFalse(any(p.requires_grad for p in visual.blocks[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in visual.blocks[-1].parameters()))
        with self.assertRaises(ValueError):
            configure_training(model, TrainingPolicy())

    def test_real_lora_loss_and_no_seg_distributed_participation(self):
        for bridge in ("single", "four_query"):
            for counts in ((1, 0, 2), (0, 0, 0)):
                model = self.fixture.model(bridge)
                policy = TrainingPolicy(lora_dropout=0)
                groups, manifest = configure_training(model, policy)
                model.gradient_checkpointing_enable()
                batch = self.batch(counts)
                output = model(**batch)
                losses = compute_losses(output, batch, policy)
                self.assertTrue(torch.isfinite(losses["loss"]))
                losses["loss"].backward()
                missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
                self.assertEqual(missing, [])
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
                self.assertEqual(float(losses["switch_token_count"]), 3)
                if sum(counts):
                    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                    optimizer = torch.optim.AdamW(groups, weight_decay=0)
                    optimizer.step()
                    for group in manifest:
                        self.assertTrue(any(not torch.equal(before[n], dict(model.named_parameters())[n])
                                            for n in group["parameter_names"]), group["name"])

    def test_mixed_diagnostic_selects_existing_samples_without_relabeling(self):
        from train_qwen_sam import diagnostic_pool
        from types import SimpleNamespace
        samples = [SimpleNamespace(is_rail_reasoning=False, masks=[1]),
                   SimpleNamespace(is_rail_reasoning=True, masks=[1]),
                   SimpleNamespace(is_rail_reasoning=True, masks=[])]
        class Dataset:
            def __init__(self):
                self.index = 0
            def __getitem__(self, index):
                sample = samples[self.index % 3]
                self.index += 1
                return sample
        for actual, expected in zip(diagnostic_pool(Dataset()), samples):
            self.assertIs(actual, expected)

    def test_weighted_ce_shift_and_chunk_invariance(self):
        batch = self.batch()
        batch["ce_token_weights"][batch["labels"].eq(60)] = 60
        model = self.fixture.model().eval()
        output = model(**batch)
        result = compute_losses(output, batch, TrainingPolicy(), 1)
        logits, labels = output["logits"][:, :-1], batch["labels"][:, 1:]
        weights = batch["ce_token_weights"][:, 1:]
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), labels.reshape(-1),
            reduction="none", ignore_index=-100).reshape_as(labels)
        expected = (ce * weights).sum() / weights[labels.ne(-100)].sum()
        torch.testing.assert_close(result["ce_loss"], expected)
        torch.testing.assert_close(result["loss"], compute_losses(output, batch, TrainingPolicy(), 128)["loss"])

    def test_mask_losses_retain_original_formulas(self):
        # Execute the original definitions from Git, without importing LLaVA.
        import ast
        import subprocess
        source = subprocess.check_output(["git", "show", "HEAD:model/LISA.py"], text=True)
        tree = ast.parse(source)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("dice_loss", "sigmoid_ce_loss")]
        namespace = {"torch": torch, "F": torch.nn.functional}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_losses", "exec"), namespace)
        prediction, target, maps = torch.randn(2, 5, 6), torch.randint(0, 2, (2, 5, 6)).float(), torch.rand(2, 5, 6)
        torch.testing.assert_close(dice_loss(prediction, target, 2), namespace["dice_loss"](prediction, target, 2))
        torch.testing.assert_close(sigmoid_ce_loss(prediction, target, 2, 1, 2, maps),
                                  namespace["sigmoid_ce_loss"](prediction, target, 2, 1, 2, maps))

    def test_full_state_restores_every_component_and_optimizer(self):
        policy = TrainingPolicy(lora_dropout=0)
        model = self.fixture.model("four_query")
        groups, _ = configure_training(model, policy)
        optimizer = torch.optim.AdamW(groups)
        batch = self.batch()
        compute_losses(model(**batch), batch, policy)["loss"].backward()
        optimizer.step()
        restored = self.fixture.model("four_query")
        new_groups, _ = configure_training(restored, policy)
        restored.load_state_dict(deepcopy(model.state_dict()), strict=True)
        new_optimizer = torch.optim.AdamW(new_groups)
        new_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
        for name, parameter in restored.named_parameters():
            torch.testing.assert_close(parameter, dict(model.named_parameters())[name])
        model.eval(); restored.eval()
        torch.testing.assert_close(model(**batch)["pred_masks"][0], restored(**batch)["pred_masks"][0])
        self.assertEqual(len(optimizer.state), len(new_optimizer.state))

    def test_checkpoint_architecture_reconstruction_needs_no_original_weights(self):
        from unittest.mock import patch
        model = self.fixture.model("four_query")
        policy = TrainingPolicy(lora_dropout=0)
        _, manifest = configure_training(model, policy)
        model.processor.save_pretrained = lambda path: Path(path).mkdir(parents=True)
        with tempfile.TemporaryDirectory() as temp:
            save_run_metadata(temp, model, policy, {"epochs": 20}, manifest)
            with patch("utils.qwen_checkpoint.AutoProcessor.from_pretrained", return_value=model.processor), \
                 patch("utils.qwen_checkpoint.build_sam_vit_h", side_effect=lambda **kwargs: fixtures.tiny_sam()):
                restored, _, _ = reconstruct_training_model(temp, torch.float32)
            restored.load_state_dict(model.state_dict(), strict=True)
            restored.validate_auxiliary_parameters()
            self.assertEqual(restored.config, model.config)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value)

    def test_full_checkpoint_rejects_bridge_legacy_missing_gate_fallback(self):
        model = self.fixture.model("four_query")
        state = dict(model.state_dict())
        state.pop("seg_query_bridge.residual_gate")
        with self.assertRaisesRegex(ValueError, "residual_gate"):
            assert_complete_state(model, state)

    def test_retention_only_prunes_owned_epochs_and_keeps_latest_best(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "training_config.json").write_text(json.dumps({"schema_version": 1,
                "checkpoint_type": "deepspeed_zero2_full"}))
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            for name in ("epoch_001", "epoch_002", "epoch_003", "epoch_004", "unrelated"):
                (checkpoints / name).mkdir()
            (checkpoints / "latest").write_text("epoch_004")
            self.assertEqual(prune_completed_checkpoints(root, 2, ["epoch_001"]), ["epoch_002"])
            self.assertTrue((checkpoints / "epoch_001").is_dir())
            self.assertTrue((checkpoints / "epoch_004").is_dir())
            self.assertTrue((checkpoints / "unrelated").is_dir())

    def test_annotation_failure_does_not_report_optimistic_global_iou(self):
        summary = summarize_test_results([
            {"image": "a", "has_mask_label": True, "union": 10, "intersection": 10, "iou": 1},
            {"image": "b", "has_mask_label": True, "error": "annotation missing"}])
        self.assertEqual(summary["giou"], 0.5)
        self.assertIsNone(summary["ciou"])
        self.assertEqual(summary["annotation_error_count"], 1)

    def test_lora_merge_and_native_reload_preserve_trained_seg_rows(self):
        model = self.fixture.model().eval()
        configure_training(model, TrainingPolicy(lora_dropout=0))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(0, 0.005)
            model.native_qwen.get_input_embeddings().seg_row.fill_(0.01)
            model.native_qwen.get_output_embeddings().seg_row.fill_(0.02)
        batch = self.batch()
        before = model(**batch)
        model.qwen = model.qwen.merge_and_unload(safe_merge=True)
        after = model(**batch)
        torch.testing.assert_close(before["logits"], after["logits"], atol=2e-6, rtol=2e-5)
        with tempfile.TemporaryDirectory() as temp:
            with materialized_seg_rows(model.qwen):
                model.qwen.save_pretrained(temp)
            native = Qwen3VLForConditionalGeneration.from_pretrained(temp, local_files_only=True)
            torch.testing.assert_close(native.get_input_embeddings().weight[60], torch.full((32,), 0.01))
            torch.testing.assert_close(native.get_output_embeddings().weight[60], torch.full((32,), 0.02))

    def test_explicit_resume_defaults_and_schedule(self):
        args = parse_args(["--run_dir=x", "--dataset_dir=y", "--resume"])
        self.assertEqual(args.val_dataset, "ReasonSeg|val")
        self.assertEqual(args.batch_size * args.grad_accumulation_steps * 8, 16)
        self.assertFalse(args.zero_offload_optimizer)
        config = deepspeed_config(1, 2, 8, TrainingPolicy(), 1000)
        self.assertEqual(config["zero_optimization"]["stage"], 2)
        self.assertNotIn("offload_optimizer", config["zero_optimization"])
        self.assertEqual(config["scheduler"]["params"]["warmup_max_lr"], [1e-4, 1e-5, 1e-4, 1e-4])
        self.assertIn("offload_optimizer", deepspeed_config(1, 2, 8, TrainingPolicy(), 1000, True)["zero_optimization"])
        check_resume_settings({"settings": {"epochs": 20}}, {"epochs": 20})
        with self.assertRaisesRegex(ValueError, "epochs"):
            check_resume_settings({"settings": {"epochs": 20}}, {"epochs": 21})

    def test_missing_masks_and_errors_remain_in_denominators(self):
        records = [{"image": "a", "prediction": "This is a turnout switch. The train takes the left route.",
                    "expected": {"switch": "T", "direction": "L"}, "mask_count": 1,
                    "has_mask_label": True, "intersection": 10, "union": 10, "iou": 1.0},
                   {"image": "b", "error": "missing", "expected": {"switch": "T", "direction": "L"},
                    "mask_count": 0, "has_mask_label": True, "intersection": 0, "union": 10}]
        summary = summarize_test_results(records)
        self.assertEqual(summary["giou"], 0.5)
        self.assertAlmostEqual(summary["ciou"], 0.5)
        self.assertEqual(summary["switch_type_accuracy"], 0.5)
        self.assertEqual(summary["one_mask_rate"], 0.5)

    def test_manifest_preserves_nonexistent_requested_image_and_tag_safety(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "manifest.json"
            path.write_text(json.dumps([{"image_path": "absent.jpg"}]))
            self.assertEqual(load_requests(manifest_path=path)[0]["image_path"], str((root / "absent.jpg").resolve()))
            with self.assertRaises(ValueError):
                resolve_checkpoint_tag(root, "../escape")

    def test_device_transfer_does_not_cast_labels_or_weight_maps(self):
        batch = self.batch()
        moved = move_batch(batch, "cpu")
        self.assertEqual(moved["images"].dtype, torch.bfloat16)
        self.assertEqual(moved["labels"].dtype, torch.int64)
        self.assertEqual(moved["ce_token_weights"].dtype, torch.float32)

    def test_export_probe_rejects_changed_answers_and_excess_mask_drift(self):
        from merge_qwen_sam import compare_probes
        baseline = {"answers": ["[SEG]"], "generated_masks": [torch.ones(1, 2, 2)],
                    "teacher_masks": [torch.ones(1, 2, 2)], "logits": torch.zeros(1, 2, 64)}
        self.assertEqual(compare_probes(baseline, deepcopy(baseline), 0.03)["max_abs_difference"], 0)
        other = deepcopy(baseline)
        other["answers"] = ["changed"]
        with self.assertRaises(ValueError):
            compare_probes(baseline, other, 0.03)
        other = deepcopy(baseline)
        baseline["generated_masks"][0][0, 0, 0] = 0.0001
        other["generated_masks"][0][0, 0, 0] = -0.0001
        with self.assertRaisesRegex(ValueError, "disagreement"):
            compare_probes(baseline, other, 0.03)

    def test_bf16_masks_are_saved_losslessly_as_binary_png(self):
        from utils.qwen_evaluation import save_binary_mask
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mask.png"
            save_binary_mask(torch.tensor([[1, -1], [0, 2]], dtype=torch.bfloat16), path)
            self.assertEqual(np.asarray(Image.open(path)).tolist(), [[255, 0], [0, 255]])
            with self.assertRaises(FloatingPointError):
                save_binary_mask(torch.tensor([[float("nan")]]), path)


if __name__ == "__main__":
    unittest.main()
