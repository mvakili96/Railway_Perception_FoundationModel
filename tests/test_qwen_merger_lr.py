"""Separate visual-merger rates, scheduler alignment and old checkpoint loading."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ML_ERROR = None
try:
    import torch
    import test_qwen_sam as fixtures
    from utils.qwen_training import TrainingPolicy, configure_training, deepspeed_config
    from utils.qwen_checkpoint import check_resume_settings, reconstruct_training_model, save_run_metadata
    from train_qwen_sam import parse_args
except ModuleNotFoundError as exc:
    ML_ERROR = str(exc)


@unittest.skipIf(ML_ERROR, "Qwen dependencies unavailable: " + str(ML_ERROR))
class QwenMergerLRTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.fixture = fixtures.CompositeModelTest()

    def test_cli_default_and_explicit_merger_rate(self):
        arguments = ["--run_dir=x", "--dataset_dir=y", "--resume"]
        default = parse_args(arguments)
        self.assertEqual(default.merger_lr, 1e-5)
        tuned = parse_args(arguments + ["--merger_lr=0.0001"])
        self.assertEqual(tuned.merger_lr, 1e-4)
        self.assertEqual(tuned.vision_lr, default.vision_lr)
        self.assertEqual(tuned.lr, default.lr)
        self.assertEqual(tuned.use_rail_reasoning_weighted_ce, default.use_rail_reasoning_weighted_ce)
        self.assertEqual(tuned.train_sam_last_blocks, default.train_sam_last_blocks)

    def test_rates_must_be_positive_and_finite(self):
        for field in ("lr", "lora_lr", "vision_lr", "merger_lr"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    TrainingPolicy(**{field: value})

    def test_exact_merger_membership_and_disjoint_complete_partition(self):
        model = self.fixture.model("four_query")
        groups, manifest = configure_training(model, TrainingPolicy(vision_last_blocks=2, merger_lr=1e-4))
        indexed = {group["name"]: group for group in manifest}
        self.assertEqual([group["lr"] for group in groups], [1e-4, 1e-5, 1e-4, 1e-4, 1e-4])
        names = [name for group in manifest for name in group["parameter_names"]]
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(set(names), trainable)
        self.assertEqual(len(names), len(set(names)))
        expected_mergers = {name for name in trainable if
                            ".visual.merger." in name or ".visual.deepstack_merger_list." in name}
        self.assertEqual(set(indexed["vision_mergers"]["parameter_names"]), expected_mergers)
        self.assertTrue(any(".visual.merger." in name for name in expected_mergers))
        self.assertTrue(any(".visual.deepstack_merger_list." in name for name in expected_mergers))
        backbone = indexed["vision_backbone"]["parameter_names"]
        self.assertTrue(backbone)
        self.assertTrue(all(".visual.blocks." in name for name in backbone))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.native_qwen.model.visual.blocks[0].parameters()))
        parameters = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))

    def test_changing_merger_rate_leaves_all_other_groups_unchanged(self):
        first = self.fixture.model()
        second = self.fixture.model()
        _, before = configure_training(first, TrainingPolicy(vision_last_blocks=2))
        _, after = configure_training(second, TrainingPolicy(vision_last_blocks=2, merger_lr=1e-4))
        for old, new in zip(before, after):
            self.assertEqual(old["parameter_names"], new["parameter_names"])
            self.assertEqual(old["parameter_count"], new["parameter_count"])
            if old["name"] == "vision_mergers":
                self.assertEqual((old["lr"], new["lr"]), (1e-5, 1e-4))
            else:
                self.assertEqual(old, new)

    def test_adamw_applies_the_independent_rates(self):
        model = self.fixture.model()
        groups, _ = configure_training(model, TrainingPolicy(vision_last_blocks=2, merger_lr=1e-4))
        representatives = []
        with torch.no_grad():
            for group in groups:
                parameter = group["params"][0]
                parameter.zero_()
                parameter.grad = torch.ones_like(parameter)
                representatives.append((parameter, group["lr"]))
        optimizer = torch.optim.AdamW(groups, weight_decay=0)
        optimizer.step()
        for parameter, rate in representatives:
            torch.testing.assert_close(parameter, torch.full_like(parameter, -rate))

    def test_scheduler_follows_actual_groups_including_frozen_backbone(self):
        for count in (0, 2):
            with self.subTest(vision_last_blocks=count):
                model = self.fixture.model()
                policy = TrainingPolicy(vision_last_blocks=count, merger_lr=1e-4)
                groups, _ = configure_training(model, policy)
                config = deepspeed_config(1, 2, 8, policy, 1000, optimizer_groups=groups)
                scheduler = config["scheduler"]["params"]
                self.assertEqual(scheduler["warmup_max_lr"], [group["lr"] for group in groups])
                self.assertEqual(scheduler["warmup_min_lr"], [0.0] * len(groups))
                self.assertEqual(scheduler["warmup_num_steps"], 100)
                self.assertEqual(config["train_batch_size"], 16)
                self.assertEqual(len(groups), 5 if count else 4)
                self.assertIn("vision_mergers", [group["group_name"] for group in groups])

    def test_legacy_group_keeps_its_original_order_and_rate(self):
        model = self.fixture.model()
        policy = TrainingPolicy(vision_last_blocks=2, vision_lr=3e-6, merger_lr=3e-6)
        groups, manifest = configure_training(model, policy, legacy_vision_group=True)
        self.assertEqual([group["group_name"] for group in groups],
                         ["language_lora", "vision", "bridge_and_seg", "sam"])
        self.assertEqual(groups[1]["lr"], 3e-6)
        self.assertTrue(any("deepstack_merger_list" in name for name in manifest[1]["parameter_names"]))
        scheduler = deepspeed_config(1, 2, 8, policy, 1000, optimizer_groups=groups)["scheduler"]["params"]
        self.assertEqual(scheduler["warmup_max_lr"], [1e-4, 3e-6, 1e-4, 1e-4])
        with self.assertRaisesRegex(ValueError, "Legacy vision groups"):
            configure_training(self.fixture.model(), TrainingPolicy(merger_lr=1e-4), legacy_vision_group=True)

    def test_new_and_legacy_runs_reconstruct_exact_weights_and_optimizer_state(self):
        for legacy, count in ((False, 0), (False, 2), (True, 0), (True, 2)):
            with self.subTest(legacy=legacy, vision_last_blocks=count), tempfile.TemporaryDirectory() as temp:
                model = self.fixture.model("four_query")
                policy = TrainingPolicy(vision_last_blocks=count, vision_lr=3e-6,
                                        merger_lr=3e-6 if legacy else 1e-4, lora_dropout=0)
                groups, manifest = configure_training(model, policy, legacy_vision_group=legacy)
                optimizer = torch.optim.AdamW(groups, weight_decay=0)
                for group in groups:
                    for parameter in group["params"]:
                        parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                model.processor.save_pretrained = lambda path: Path(path).mkdir(parents=True)
                settings = {"epochs": 20, "vision_lr": policy.vision_lr, "merger_lr": policy.merger_lr}
                save_run_metadata(temp, model, policy, settings, manifest)
                if legacy:
                    metadata_path = Path(temp) / "training_config.json"
                    old = json.loads(metadata_path.read_text())
                    old["policy"].pop("merger_lr")
                    old["settings"].pop("merger_lr")
                    metadata_path.write_text(json.dumps(old))
                with patch("utils.qwen_checkpoint.AutoProcessor.from_pretrained", return_value=model.processor), \
                     patch("utils.qwen_checkpoint.build_sam_vit_h", side_effect=lambda **kwargs: fixtures.tiny_sam()):
                    restored, restored_groups, metadata = reconstruct_training_model(temp, torch.float32)
                restored.load_state_dict(model.state_dict(), strict=True)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value)
                restored_optimizer = torch.optim.AdamW(restored_groups, weight_decay=0)
                restored_optimizer.load_state_dict(optimizer.state_dict())
                self.assertEqual([group["lr"] for group in restored_optimizer.param_groups],
                                 [group["lr"] for group in optimizer.param_groups])
                self.assertEqual(len(restored_optimizer.state), len(optimizer.state))
                for original, reconstructed in zip(optimizer.state.values(), restored_optimizer.state.values()):
                    self.assertEqual(original.keys(), reconstructed.keys())
                    for key in original:
                        torch.testing.assert_close(original[key], reconstructed[key])
                self.assertEqual("merger_lr" in metadata["policy"], not legacy)
                check_resume_settings(metadata, settings)
                for key in ("merger_lr", "vision_lr", "epochs"):
                    changed = {**settings, key: settings[key] * 2}
                    with self.subTest(changed=key), self.assertRaisesRegex(ValueError, key):
                        check_resume_settings(metadata, changed)

    def test_new_metadata_cannot_silently_omit_merger_lr(self):
        metadata = {"policy": asdict(TrainingPolicy()), "settings": {"vision_lr": 1e-5}}
        with self.assertRaisesRegex(ValueError, "merger_lr"):
            check_resume_settings(metadata, {"vision_lr": 1e-5, "merger_lr": 1e-5})

    def test_launcher_exposes_rates_and_preserves_user_overrides(self):
        launcher = (Path(__file__).resolve().parents[1] / "fine_tune_Qwen_2nodes.sbatch").read_text()
        self.assertIn("--lr=0.0001", launcher)
        self.assertIn("--vision_lr=0.00001", launcher)
        self.assertIn("--merger_lr=0.00001", launcher)
        self.assertIn('"${initialization[@]}" "$@"', launcher)
        self.assertIn('/scratch/mohammadjavad/QWEN-LISA_Codebase', launcher)
        # The same argument ordering used by sbatch lets an explicit submission
        # override its experiment default without changing any other rates.
        arguments = ["--run_dir=x", "--dataset_dir=y", "--resume", "--lr=0.0001",
                     "--vision_lr=0.00001", "--merger_lr=0.0001", "--merger_lr=0.00003"]
        args = parse_args(arguments)
        self.assertEqual((args.lr, args.vision_lr, args.merger_lr), (1e-4, 1e-5, 3e-5))


if __name__ == "__main__":
    unittest.main()
