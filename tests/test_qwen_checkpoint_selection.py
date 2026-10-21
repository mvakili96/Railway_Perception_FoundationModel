"""Best-checkpoint decisions, metric labels, resume safety and retention."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from utils.qwen_checkpoint_selection import (BEST_CHECKPOINT_METRICS, DEFAULT_BEST_CHECKPOINT_METRIC,
    best_checkpoint_record, restore_best_checkpoint, select_best_checkpoint)
from utils.rail_switch_validation import summarize_switch_validation, switch_validation_wandb_metrics


JOINT = "val/switch_subset/joint_accuracy"


class CheckpointSelectionTest(unittest.TestCase):
    def test_default_and_supported_criteria(self):
        self.assertEqual(DEFAULT_BEST_CHECKPOINT_METRIC, "val/giou")
        self.assertEqual(BEST_CHECKPOINT_METRICS, ("val/giou", "val/ciou", JOINT))

    def test_each_criterion_picks_its_own_best_epoch(self):
        epochs = [
            {"val/giou": 0.95, "val/ciou": 0.97, JOINT: 0.5},
            {"val/giou": 0.80, "val/ciou": 0.82, JOINT: 0.9},
            {"val/giou": 0.97, "val/ciou": 0.96, JOINT: 0.8},
        ]
        for metric, expected in (("val/giou", (0.97, "epoch_003")),
                                 ("val/ciou", (0.97, "epoch_001")),
                                 (JOINT, (0.9, "epoch_002"))):
            with self.subTest(metric=metric):
                score, tag = -1.0, None
                for index, metrics in enumerate(epochs, 1):
                    score, tag = select_best_checkpoint(metrics, metric, score, tag, f"epoch_{index:03d}")
                self.assertEqual((score, tag), expected)

    def test_ties_keep_the_earlier_best_checkpoint(self):
        self.assertEqual(select_best_checkpoint({JOINT: 0.85}, JOINT, 0.85, "epoch_002", "epoch_003"),
                         (0.85, "epoch_002"))

    def test_zero_is_a_valid_initial_accuracy(self):
        self.assertEqual(select_best_checkpoint({JOINT: 0.0}, JOINT, -1.0, None, "epoch_001"),
                         (0.0, "epoch_001"))

    def test_missing_selected_metric_never_falls_back_to_giou(self):
        with self.assertRaisesRegex(ValueError, "not computed"):
            select_best_checkpoint({"val/giou": 0.99}, JOINT, -1.0, None, "epoch_001")
        self.assertEqual(select_best_checkpoint({}, "val/giou", -1.0, None, "epoch_001", allow_missing=True),
                         (-1.0, None))

    def test_nonfinite_or_invalid_scores_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), None, "invalid"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_best_checkpoint({JOINT: value}, JOINT, -1.0, None, "epoch_001")

    def test_unsupported_metric_is_rejected_even_without_evaluation(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            select_best_checkpoint({}, "train/loss", -1.0, None, "epoch_001", allow_missing=True)

    def test_records_label_joint_accuracy_without_calling_it_giou(self):
        record = best_checkpoint_record(JOINT, 0.85, "epoch_002")
        self.assertEqual(record, {"tag": "epoch_002", "metric": JOINT, "score": 0.85, "joint_accuracy": 0.85})
        self.assertNotIn("giou", record)
        self.assertEqual(json.loads(json.dumps(record, allow_nan=False)), record)

    def test_giou_records_preserve_the_legacy_score_alias(self):
        record = best_checkpoint_record("val/giou", 0.91, "epoch_004")
        self.assertEqual(record["giou"], 0.91)
        self.assertEqual(record["score"], 0.91)
        self.assertEqual(record["metric"], "val/giou")
        self.assertEqual(best_checkpoint_record("val/ciou", 0.92, "epoch_004")["ciou"], 0.92)

    def test_no_eval_has_no_fake_best_score(self):
        self.assertEqual(best_checkpoint_record("val/giou", -1.0, None),
                         {"tag": None, "metric": "val/giou", "score": None, "giou": None})

    def test_legacy_resume_means_giou_only(self):
        client = {"best_score": 0.9, "best_tag": "epoch_002"}
        self.assertEqual(restore_best_checkpoint(client, "val/giou"), (0.9, "epoch_002"))
        with self.assertRaisesRegex(ValueError, "Cannot change"):
            restore_best_checkpoint(client, JOINT)

    def test_new_resume_keeps_the_named_metric_and_rejects_switching(self):
        client = {"best_score": 0.85, "best_tag": "epoch_003", "best_checkpoint_metric": JOINT}
        self.assertEqual(restore_best_checkpoint(client, JOINT), (0.85, "epoch_003"))
        with self.assertRaisesRegex(ValueError, "Cannot change"):
            restore_best_checkpoint(client, "val/giou")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            restore_best_checkpoint({**client, "best_score": float("nan")}, JOINT)

    def test_joint_metric_is_the_same_full_denominator_metric_logged_to_wandb(self):
        manifest = [{"image": "a", "switch": "T", "direction": "L"},
                    {"image": "b", "switch": "T", "direction": "R"},
                    {"image": "c", "switch": "M", "direction": "L"},
                    {"image": "d", "switch": "M", "direction": "R"}]
        records = [{"image": "a", "prediction": "This is a turnout switch. The train takes the left route."},
                   {"image": "b", "prediction": "This is a turnout switch. The train takes the left route."},
                   {"image": "c", "prediction": "This is a turnout switch. The train takes the left route."},
                   {"image": "d", "error": "generation failed"}]
        metrics = switch_validation_wandb_metrics(summarize_switch_validation(manifest, records))
        self.assertEqual(metrics[JOINT], 0.25)
        self.assertEqual(select_best_checkpoint(metrics, JOINT, -1.0, None, "epoch_001"), (0.25, "epoch_001"))


ML_ERROR = None
try:
    from train_qwen_sam import parse_args
    from utils.qwen_checkpoint import check_resume_settings, prune_completed_checkpoints
except ModuleNotFoundError as exc:
    ML_ERROR = str(exc)


@unittest.skipIf(ML_ERROR, "Qwen dependencies unavailable: " + str(ML_ERROR))
class CheckpointSelectionIntegrationTest(unittest.TestCase):
    def test_cli_defaults_and_explicit_choices(self):
        arguments = ["--run_dir=x", "--dataset_dir=y", "--resume"]
        self.assertEqual(parse_args(arguments).best_checkpoint_metric, "val/giou")
        for metric in BEST_CHECKPOINT_METRICS:
            with self.subTest(metric=metric):
                self.assertEqual(parse_args(arguments + ["--best_checkpoint_metric=" + metric]).best_checkpoint_metric,
                                 metric)

    def test_joint_selection_requires_generation_validation(self):
        arguments = ["--run_dir=x", "--dataset_dir=y", "--resume", "--best_checkpoint_metric=" + JOINT]
        for option in ("--no_eval", "--no-epoch_switch_validation"):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(arguments + [option])
        # Existing no-evaluation smoke/gate workflows remain supported.
        self.assertTrue(parse_args(["--run_dir=x", "--dataset_dir=y", "--resume", "--no_eval"]).no_eval)

    def test_legacy_settings_resume_only_under_the_original_criterion(self):
        metadata = {"settings": {"epochs": 20}}
        original = deepcopy(metadata)
        settings = {"epochs": 20, "best_checkpoint_metric": "val/giou"}
        check_resume_settings(metadata, settings)
        self.assertEqual(metadata, original)
        with self.assertRaisesRegex(ValueError, "best_checkpoint_metric"):
            check_resume_settings(metadata, {**settings, "best_checkpoint_metric": JOINT})
        new = {"settings": {**settings, "best_checkpoint_metric": JOINT}}
        check_resume_settings(new, new["settings"])
        with self.assertRaisesRegex(ValueError, "best_checkpoint_metric"):
            check_resume_settings(new, settings)

    def test_retention_protects_joint_best_even_when_giou_prefers_another_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "training_config.json").write_text(json.dumps({"schema_version": 1,
                "checkpoint_type": "deepspeed_zero2_full"}))
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            score, tag = -1.0, None
            for index, joint in enumerate((0.5, 0.9, 0.8, 0.7, 0.6), 1):
                epoch_tag = f"epoch_{index:03d}"
                (checkpoints / epoch_tag).mkdir()
                score, tag = select_best_checkpoint({JOINT: joint, "val/giou": 0.8 + index * 0.01},
                                                    JOINT, score, tag, epoch_tag)
            (checkpoints / "latest").write_text("epoch_005")
            self.assertEqual(tag, "epoch_002")
            self.assertEqual(prune_completed_checkpoints(root, 2, [tag]), ["epoch_001", "epoch_003"])
            self.assertEqual(sorted(path.name for path in checkpoints.iterdir() if path.is_dir()),
                             ["epoch_002", "epoch_004", "epoch_005"])

    def test_launcher_exposes_the_metric_and_allows_an_override(self):
        launcher = (Path(__file__).resolve().parents[1] / "fine_tune_Qwen_2nodes.sbatch").read_text()
        self.assertIn("--best_checkpoint_metric=val/giou", launcher)
        self.assertIn(JOINT, launcher)
        self.assertIn('"${initialization[@]}" "$@"', launcher)
        args = parse_args(["--run_dir=x", "--dataset_dir=y", "--resume", "--best_checkpoint_metric=val/giou",
                           "--best_checkpoint_metric=" + JOINT])
        self.assertEqual(args.best_checkpoint_metric, JOINT)


if __name__ == "__main__":
    unittest.main()
