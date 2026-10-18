"""Language-only learning-rate tuning and historical checkpoint compatibility."""

from dataclasses import asdict
import json
from pathlib import Path
import re
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
class QwenLoRALRTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.fixture = fixtures.CompositeModelTest()

    def test_cli_exposes_independent_lora_rate_without_changing_other_defaults(self):
        arguments = ["--run_dir=x", "--dataset_dir=y", "--resume"]
        default = parse_args(arguments)
        self.assertEqual(default.lora_lr, 1e-4)
        tuned = parse_args(arguments + ["--lora_lr=0.00003"])
        self.assertEqual(tuned.lora_lr, 3e-5)
        self.assertEqual({key: value for key, value in vars(default).items() if key != "lora_lr"},
                         {key: value for key, value in vars(tuned).items() if key != "lora_lr"})

    def test_lora_rate_must_be_finite_and_positive(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TrainingPolicy(lora_lr=value)

    def test_only_language_lora_group_changes(self):
        for bridge in ("single", "four_query"):
            with self.subTest(bridge=bridge):
                baseline = self.fixture.model(bridge)
                tuned = self.fixture.model(bridge)
                _, before = configure_training(baseline, TrainingPolicy(vision_last_blocks=2))
                groups, after = configure_training(tuned, TrainingPolicy(vision_last_blocks=2, lora_lr=3e-5))
                self.assertEqual([group["lr"] for group in groups], [3e-5, 1e-5, 1e-5, 1e-4, 1e-4])
                self.assertEqual(len(before), len(after))
                for old, new in zip(before, after):
                    if old["name"] == "language_lora":
                        self.assertEqual((old["lr"], new["lr"]), (1e-4, 3e-5))
                        self.assertEqual({key: value for key, value in old.items() if key != "lr"},
                                         {key: value for key, value in new.items() if key != "lr"})
                        self.assertTrue(all(".language_model.layers." in name and ".lora_" in name
                                            for name in new["parameter_names"]))
                    else:
                        self.assertEqual(old, new)
                names = [name for group in after for name in group["parameter_names"]]
                expected = {name for name, parameter in tuned.named_parameters() if parameter.requires_grad}
                self.assertEqual(set(names), expected)
                self.assertEqual(len(names), len(set(names)))

    def test_adamw_keeps_sam_bridge_and_seg_updates_at_the_main_rate(self):
        model = self.fixture.model()
        groups, _ = configure_training(model, TrainingPolicy(vision_last_blocks=2, lora_lr=3e-5))
        examples = []
        with torch.no_grad():
            for group in groups:
                parameter = group["params"][0]
                parameter.zero_()
                parameter.grad = torch.ones_like(parameter)
                examples.append((parameter, group["lr"]))
            for module in (model.native_qwen.get_input_embeddings(), model.native_qwen.get_output_embeddings()):
                module.seg_row.zero_()
                module.seg_row.grad = torch.ones_like(module.seg_row)
                examples.append((module.seg_row, 1e-4))
        torch.optim.AdamW(groups, weight_decay=0).step()
        for parameter, rate in examples:
            torch.testing.assert_close(parameter, torch.full_like(parameter, -rate))

    def test_scheduler_uses_lora_rate_in_split_and_legacy_groups(self):
        policy = TrainingPolicy(lora_lr=3e-5)
        fallback = deepspeed_config(1, 2, 8, policy, 1000)["scheduler"]["params"]
        self.assertEqual(fallback["warmup_max_lr"], [3e-5, 1e-5, 1e-5, 1e-4, 1e-4])
        for legacy, count in ((False, 0), (False, 2), (True, 0), (True, 2)):
            with self.subTest(legacy=legacy, vision_last_blocks=count):
                model = self.fixture.model()
                policy = TrainingPolicy(lora_lr=3e-5, vision_last_blocks=count)
                groups, _ = configure_training(model, policy, legacy_vision_group=legacy)
                config = deepspeed_config(1, 2, 8, policy, 1000, optimizer_groups=groups)
                scheduler = config["scheduler"]["params"]
                self.assertEqual(scheduler["warmup_max_lr"], [group["lr"] for group in groups])
                self.assertEqual(scheduler["warmup_min_lr"], [0.0] * len(groups))
                self.assertEqual(scheduler["warmup_max_lr"][0], 3e-5)
                self.assertEqual(config["train_batch_size"], 16)
                self.assertEqual(scheduler["warmup_num_steps"], 100)

    def test_all_checkpoint_generations_restore_original_rates_and_optimizer_state(self):
        for generation in ("combined_vision", "split_vision", "independent_lora"):
            with self.subTest(generation=generation), tempfile.TemporaryDirectory() as temp:
                legacy = generation == "combined_vision"
                independent = generation == "independent_lora"
                model = self.fixture.model("four_query")
                # Non-default historical LR catches accidental fallback to 1e-4.
                policy = TrainingPolicy(lr=7e-5, lora_lr=3e-5 if independent else 7e-5,
                    vision_lr=4e-6, merger_lr=4e-6 if legacy else 8e-6,
                    vision_last_blocks=2, lora_dropout=0)
                groups, manifest = configure_training(model, policy, legacy_vision_group=legacy)
                optimizer = torch.optim.AdamW(groups, weight_decay=0)
                for group in groups:
                    for parameter in group["params"]:
                        parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                model.processor.save_pretrained = lambda path: Path(path).mkdir(parents=True)
                settings = {"epochs": 20, "lr": policy.lr, "lora_lr": policy.lora_lr,
                            "vision_lr": policy.vision_lr, "merger_lr": policy.merger_lr}
                save_run_metadata(temp, model, policy, settings, manifest)
                metadata_path = Path(temp) / "training_config.json"
                metadata = json.loads(metadata_path.read_text())
                if not independent:
                    metadata["policy"].pop("lora_lr")
                    metadata["settings"].pop("lora_lr")
                if legacy:
                    metadata["policy"].pop("merger_lr")
                    metadata["settings"].pop("merger_lr")
                metadata_path.write_text(json.dumps(metadata))
                original_metadata = metadata_path.read_text()
                with patch("utils.qwen_checkpoint.AutoProcessor.from_pretrained", return_value=model.processor), \
                     patch("utils.qwen_checkpoint.build_sam_vit_h", side_effect=lambda **kwargs: fixtures.tiny_sam()):
                    restored, restored_groups, read = reconstruct_training_model(temp, torch.float32)
                restored.load_state_dict(model.state_dict(), strict=True)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value)
                self.assertEqual([group["lr"] for group in restored_groups], [group["lr"] for group in groups])
                restored_optimizer = torch.optim.AdamW(restored_groups, weight_decay=0)
                restored_optimizer.load_state_dict(optimizer.state_dict())
                for old, new in zip(optimizer.state.values(), restored_optimizer.state.values()):
                    self.assertEqual(old.keys(), new.keys())
                    for key in old:
                        torch.testing.assert_close(old[key], new[key])
                self.assertEqual(metadata_path.read_text(), original_metadata)
                self.assertEqual(read, metadata)
                check_resume_settings(read, settings)
                # Checkpoint client settings may already contain the normalized
                # fields after an earlier resume; these must also stay usable.
                check_resume_settings({**read, "settings": settings}, settings)
                for key in ("lr", "lora_lr", "vision_lr", "merger_lr", "epochs"):
                    with self.subTest(changed=key), self.assertRaisesRegex(ValueError, key):
                        check_resume_settings(read, {**settings, key: settings[key] / 2})

    def test_new_metadata_cannot_silently_omit_lora_lr(self):
        metadata = {"policy": asdict(TrainingPolicy()), "settings": {"lr": 1e-4}}
        with self.assertRaisesRegex(ValueError, "lora_lr"):
            check_resume_settings(metadata, {"lr": 1e-4, "lora_lr": 1e-4})

    def test_launcher_selects_experiment_rates_and_allows_restoring_baseline(self):
        launcher = (Path(__file__).resolve().parents[1] / "fine_tune_Qwen_2nodes.sbatch").read_text()
        flags = re.findall(r"--(?:lr|lora_lr|vision_lr|merger_lr)=\S+", launcher)
        arguments = ["--run_dir=x", "--dataset_dir=y", "--resume"] + flags
        args = parse_args(arguments)
        self.assertEqual((args.lr, args.lora_lr, args.vision_lr, args.merger_lr),
                         (1e-4, 3e-5, 1e-5, 1e-5))
        restored = parse_args(arguments + ["--lora_lr=0.0001"])
        self.assertEqual((restored.lr, restored.lora_lr, restored.vision_lr, restored.merger_lr),
                         (1e-4, 1e-4, 1e-5, 1e-5))
        self.assertIn('"${initialization[@]}" "$@"', launcher)
        self.assertIn("--seg_prompt_bridge_type=single", launcher)
        self.assertIn("--batch_size=1 --grad_accumulation_steps=2 --epochs=20 --steps_per_epoch=50", launcher)


if __name__ == "__main__":
    unittest.main()
