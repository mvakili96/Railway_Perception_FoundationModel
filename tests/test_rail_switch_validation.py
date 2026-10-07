import ast
import contextlib
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from scripts.data.install_rail_switch_validation import install_labels
from utils.rail_switch_validation import (
    IMAGE_RELATIVE_PATH,
    SWITCH_VALIDATION_PROMPT,
    load_switch_validation_labels,
    parse_switch_validation_answer,
    resolve_switch_validation_manifest,
    summarize_switch_validation,
    switch_validation_wandb_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "scripts/data/templates/rail_switch_validation.json"


def answer(switch="turnout", direction="right"):
    return (
        "It is [SEG]. This is a {} switch. The right blade is open and "
        "the left blade is closed. Therefore, the ego-path follows the "
        "{}-hand path.".format(switch, direction)
    )


class SwitchLabelsTest(unittest.TestCase):
    def setUp(self):
        self.data = load_switch_validation_labels(SOURCE)

    def test_transcription_has_40_unique_images_and_confirmed_index(self):
        samples = self.data["samples"]
        self.assertEqual(len(samples), 40)
        self.assertEqual(sum(s["switch"] == "T" for s in samples), 20)
        self.assertEqual(sum(s["direction"] == "L" for s in samples), 24)
        self.assertIn(
            {"image_index": 8302, "image": "rs08302.jpg", "switch": "M", "direction": "R"},
            samples,
        )
        self.assertEqual(samples[0]["image"], "rs08006.jpg")
        self.assertEqual(samples[-1]["image"], "rs08496.jpg")

    def test_rejects_duplicates_bad_labels_and_shifted_filenames(self):
        mutations = (
            lambda data: data["samples"].append(data["samples"][0]),
            lambda data: data["samples"][0].update(switch="turnout"),
            lambda data: data["samples"][0].update(direction="left"),
            lambda data: data["samples"][0].update(image="rs08005.jpg"),
            lambda data: data["samples"][0].update(image_index=2305),
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "labels.json"
            for mutate in mutations:
                data = copy.deepcopy(self.data)
                mutate(data)
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_switch_validation_labels(path)

    def test_install_is_idempotent_and_preserves_manual_corrections(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp)
            images = dataset / "reason_seg/ReasonSeg/val"
            self.assertEqual(IMAGE_RELATIVE_PATH, Path("reason_seg/ReasonSeg/val"))
            images.mkdir(parents=True)
            for sample in self.data["samples"]:
                (images / sample["image"]).touch()
            destination, _ = install_labels(SOURCE, dataset)
            self.assertEqual(
                destination,
                dataset.resolve() / "reason_seg/ReasonSeg/explanatory/val_switch_labels.json",
            )
            before = destination.read_bytes()
            self.assertEqual(install_labels(SOURCE, dataset)[0], destination)
            self.assertEqual(before, destination.read_bytes())
            manifest, labels_path, image_dir = resolve_switch_validation_manifest(dataset)
            self.assertEqual(len(manifest), 40)
            self.assertEqual(labels_path, str(destination))
            self.assertEqual(image_dir, str(images.resolve()))
            edited = copy.deepcopy(self.data)
            edited["samples"][0]["direction"] = "L"
            destination.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError, "differ"):
                install_labels(SOURCE, dataset)
            self.assertEqual(json.loads(destination.read_text()), edited)
            install_labels(SOURCE, dataset, overwrite=True)
            self.assertEqual(json.loads(destination.read_text()), self.data)

    def test_missing_image_fails_before_install(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "rs08006"):
                install_labels(SOURCE, temp)
            self.assertFalse((Path(temp) / "reason_seg").exists())


class SwitchAnswerTest(unittest.TestCase):
    def test_canonical_and_case_whitespace_variations(self):
        self.assertEqual(parse_switch_validation_answer(answer()), {"switch": "T", "direction": "R"})
        self.assertEqual(
            parse_switch_validation_answer(answer("merge", "left").upper().replace(" ", "\n")),
            {"switch": "M", "direction": "L"},
        )

    def test_fields_are_independent_and_blades_do_not_determine_route(self):
        self.assertEqual(
            parse_switch_validation_answer("This is a merge switch. The right blade is open."),
            {"switch": "M", "direction": None},
        )
        self.assertEqual(
            parse_switch_validation_answer("The right blade is open. The ego-path follows the left-hand path."),
            {"switch": None, "direction": "L"},
        )
        self.assertEqual(
            parse_switch_validation_answer("The open right blade connects to the right-hand path."),
            {"switch": None, "direction": None},
        )

    def test_conflicting_decisions_and_negated_claims_are_not_accepted(self):
        parsed = parse_switch_validation_answer(answer() + " This is a merge switch. The ego-path follows the left-hand path.")
        self.assertEqual(parsed, {"switch": None, "direction": None})
        self.assertEqual(
            parse_switch_validation_answer("This is not a merge switch. The train does not take the right-hand path."),
            {"switch": None, "direction": None},
        )

    def test_summary_uses_all_labels_including_errors_missing_and_empty_answers(self):
        manifest = [
            {"image": str(i), "switch": "T", "direction": "R"} for i in range(4)
        ]
        results = [
            {"image": "0", "prediction": answer(), "mask_count": 1},
            {"image": "1", "prediction": "", "mask_count": 0},
            {"image": "2", "error": "inference failed"},
        ]
        summary = summarize_switch_validation(manifest, results)
        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["error_count"], 2)
        self.assertEqual(summary["switch_type_accuracy"], 0.25)
        self.assertEqual(summary["route_direction_accuracy"], 0.25)
        self.assertEqual(summary["one_mask_rate"], 0.25)
        metrics = switch_validation_wandb_metrics(summary)
        self.assertEqual(metrics["val/switch_subset/switch_type_accuracy"], 0.25)

    def test_duplicate_results_are_rejected(self):
        manifest = [{"image": "a", "switch": "T", "direction": "R"}]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_switch_validation(manifest, [{"image": "a"}, {"image": "a"}])


class EpochSwitchRunnerTest(unittest.TestCase):
    """Exercise runner wiring without importing CUDA/DeepSpeed on a login node.

    ML/image operations are mocked; gathering, reports, scoring, restoration,
    and W&B calls execute the real runner extracted from the training module.
    """

    def make_runner(self, temp, distributed=False):
        import os
        import traceback
        tree = ast.parse((REPO_ROOT / "train_ds.py").read_text())
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_epoch_switch_validation"
        )
        tensor = MagicMock()
        tensor.__getitem__.return_value = tensor
        tensor.lt.return_value.any.return_value = False
        torch = MagicMock()
        torch.distributed.is_available.return_value = distributed
        torch.distributed.is_initialized.return_value = distributed
        torch.distributed.get_rank.return_value = 0
        torch.distributed.get_world_size.return_value = 8
        torch.inference_mode.side_effect = contextlib.nullcontext
        cv2 = MagicMock()
        cv2.imread.return_value.shape = (2, 3, 3)
        cv2.cvtColor.return_value.shape = (2, 3, 3)
        cv2.imwrite.return_value = True
        transform = MagicMock()
        transform.apply_image.return_value.shape = (2, 3, 3)
        wandb_log = MagicMock()
        namespace = {
            "torch": torch, "cv2": cv2, "np": SimpleNamespace(uint8="uint8"),
            "os": os, "json": json, "traceback": traceback,
            "SWITCH_VALIDATION_PROMPT": SWITCH_VALIDATION_PROMPT,
            "parse_switch_validation_answer": parse_switch_validation_answer,
            "summarize_switch_validation": summarize_switch_validation,
            "switch_validation_wandb_metrics": switch_validation_wandb_metrics,
            "build_epoch_reasoning_prompt": lambda args: SWITCH_VALIDATION_PROMPT,
            "tokenizer_image_token": MagicMock(return_value=tensor),
            "ResizeLongestSide": MagicMock(return_value=transform),
            "ValDataset": SimpleNamespace(pixel_mean=MagicMock(), pixel_std=MagicMock(), img_size=1024),
            "wandb_log": wandb_log,
            "print": MagicMock(),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "train_ds.py", "exec"), namespace)
        args = SimpleNamespace(
            global_rank=0, local_rank=0, precision="bf16", image_size=1024,
            log_dir=temp, is_main_process=True, steps_per_epoch=50,
            epoch_switch_validation_max_new_tokens=256,
        )
        model = MagicMock()
        model.training = True
        model.get_base_model.return_value = model
        masks = MagicMock()
        masks.shape = (1, 2, 3)
        mask = MagicMock()
        mask.__gt__.return_value = MagicMock()
        masks.__iter__.side_effect = lambda: iter([mask])
        model.evaluate.return_value = (tensor, [masks])
        engine = MagicMock(module=model, global_steps=50)
        engine.zero_optimization_stage.return_value = 2
        tokenizer = MagicMock()
        tokenizer.decode.return_value = answer()
        return namespace["run_epoch_switch_validation"], args, engine, tokenizer, torch, wandb_log

    @staticmethod
    def manifest(count):
        return [
            {"image": "rs{:05d}.jpg".format(8006 + i), "image_index": 8006 + i,
             "image_path": "unused.jpg", "switch": "T", "direction": "R"}
            for i in range(count)
        ]

    def test_epoch_logs_iou_and_decision_metrics_together_and_restores_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            runner, args, engine, tokenizer, _, logger = self.make_runner(temp)
            summary = runner(
                engine, tokenizer, MagicMock(), self.manifest(2), "labels.json",
                0, False, args, MagicMock(), "wandb",
                val_metrics={"val/giou": 0.7, "val/ciou": 0.8},
            )
            self.assertEqual(summary["switch_type_accuracy"], 1.0)
            engine.train.assert_called_once_with(True)
            self.assertEqual(engine.module.evaluate.call_count, 2)
            self.assertFalse(engine.module.evaluate.call_args.kwargs["do_sample"])
            logger.assert_called_once()
            metrics = logger.call_args.args[1]
            self.assertEqual(metrics["val/giou"], 0.7)
            self.assertEqual(metrics["val/switch_subset/route_direction_accuracy"], 1.0)
            self.assertEqual(logger.call_args.args[2], 50)
            report = json.loads((Path(temp) / "val_switch/epoch_0001/results.json").read_text())
            self.assertEqual(len(report["results"]), 2)
            self.assertFalse(report["header"]["checkpoint_saved_before_validation"])
            self.assertEqual(report["results"][0]["mask_count"], 1)

    def test_failure_persists_all_samples_before_raising(self):
        with tempfile.TemporaryDirectory() as temp:
            runner, args, engine, tokenizer, _, logger = self.make_runner(temp)
            successful = engine.module.evaluate.return_value
            engine.module.evaluate.side_effect = [successful, RuntimeError("inference failed")]
            with self.assertRaisesRegex(RuntimeError, "Switch validation inference failed"):
                runner(
                    engine, tokenizer, MagicMock(), self.manifest(3), "labels.json",
                    0, False, args, None, "wandb",
                )
            engine.train.assert_called_once_with(True)
            report = json.loads((Path(temp) / "val_switch/epoch_0001/results.json").read_text())
            self.assertEqual(len(report["results"]), 3)
            self.assertEqual(report["summary"]["error_count"], 2)
            self.assertAlmostEqual(report["summary"]["switch_type_accuracy"], 1 / 3)
            logger.assert_called_once()

    def test_eight_rank_sharding_gathers_full_40_image_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            runner, args, engine, tokenizer, torch, _ = self.make_runner(temp, distributed=True)
            manifest = self.manifest(40)

            def gather(destination, local_results):
                destination[0] = local_results
                for rank in range(1, 8):
                    destination[rank] = [
                        {**sample, "rank": rank, "prediction": answer(), "mask_count": 1}
                        for sample in manifest[rank::8]
                    ]

            torch.distributed.all_gather_object.side_effect = gather
            summary = runner(
                engine, tokenizer, MagicMock(), manifest, "labels.json",
                0, True, args, None, "wandb",
            )
            self.assertEqual(engine.module.evaluate.call_count, 5)
            self.assertEqual(summary["result_count"], 40)
            self.assertEqual(summary["route_direction_accuracy"], 1.0)
            torch.distributed.barrier.assert_called()


if __name__ == "__main__":
    unittest.main()
