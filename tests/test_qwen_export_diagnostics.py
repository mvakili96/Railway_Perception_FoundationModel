"""Task-aware BF16 merge checks and strict reload gates preserve diagnostics."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


ML_ERROR = None
try:
    import torch
    import merge_qwen_sam as export
except ModuleNotFoundError as exc:
    ML_ERROR = str(exc)


@unittest.skipIf(ML_ERROR, "Qwen dependencies unavailable: " + str(ML_ERROR))
class QwenExportDiagnosticsTest(unittest.TestCase):
    def probe(self):
        return {"answers": ["[SEG]. This is a turnout switch."],
                "generated_masks": [torch.ones(1, 2, 2)],
                "teacher_masks": [torch.ones(1, 2, 2)], "logits": torch.zeros(1, 2, 4)}

    def compare(self, reference, candidate, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return export.compare_probes(reference, candidate, 0.03, **kwargs)

    def generation_only_probe(self):
        return {key: value for key, value in self.probe().items() if key in ("answers", "generated_masks")}

    def test_skip_flag_avoids_teacher_forward_and_default_still_runs_it(self):
        for skip in (False, True):
            with self.subTest(skip=skip):
                model = MagicMock()
                model.parameters.side_effect = lambda: iter([torch.ones(1)])
                model.generate_masks.return_value = {"answers": ["[SEG]."], "pred_masks": [torch.ones(1, 2, 2)]}
                model.return_value = {"pred_masks": [torch.ones(1, 2, 2)], "logits": torch.zeros(1, 3, 4)}
                samples = []
                def collate(items):
                    samples.extend(items)
                    return {}
                with patch.object(export, "inference_sample", side_effect=lambda *args:
                                  SimpleNamespace(original_size=(2, 2), inference=True)) as sample_factory, \
                     patch.object(export, "QwenSegCollator", return_value=collate), \
                     patch.object(export, "move_batch", side_effect=lambda batch, device, dtype: batch):
                    options = {"skip_teacher_probe": True} if skip else {}
                    result = export.export_probe(model, "/image.jpg", 256, **options)
                model.generate_masks.assert_called_once_with({}, max_new_tokens=256, do_sample=False)
                self.assertEqual(model.call_count, 0 if skip else 1)
                self.assertEqual(sample_factory.call_count, 1 if skip else 2)
                self.assertEqual(len(samples), 1 if skip else 2)
                self.assertEqual(result["answers"], ["[SEG]."])
                self.assertEqual("teacher_masks" in result, not skip)
                self.assertEqual("logits" in result, not skip)
                if not skip:
                    self.assertFalse(samples[-1].inference)
                    self.assertEqual(samples[-1].answers, ["[SEG]."])
                    self.assertEqual(result["logits"].shape, (1, 2, 4))

    def test_generation_only_comparisons_record_skipped_teacher_checks(self):
        for policy in ("merge", "strict"):
            with self.subTest(policy=policy):
                report = self.compare(self.generation_only_probe(), self.generation_only_probe(),
                                      policy=policy, skip_teacher_probe=True)
                self.assertTrue(report["passed"])
                self.assertTrue(report["teacher_probe_skipped"])
                self.assertEqual([entry["name"] for entry in report["tensors"]], ["generated_masks[0]"])
                self.assertEqual(report["tensors"][0]["raw_logit_gate_applied"], policy == "strict")

    def test_teacher_data_cannot_disappear_without_explicit_skip(self):
        with self.assertRaisesRegex(export.ProbeComparisonError, "Missing required") as raised:
            self.compare(self.generation_only_probe(), self.generation_only_probe(), policy="merge")
        self.assertFalse(raised.exception.report["teacher_probe_skipped"])
        self.assertEqual(len(raised.exception.report["failures"]), 2)

    def test_skip_keeps_generated_answer_and_mask_safety_gates(self):
        for change in ("answer", "mask", "nonfinite", "dtype", "shape", "missing_seg", "multiple_seg"):
            before, after = self.generation_only_probe(), self.generation_only_probe()
            if change == "answer":
                after["answers"] = ["A different answer."]
            elif change == "mask":
                after["generated_masks"][0][0, 0, 0] = -1
            elif change == "nonfinite":
                after["generated_masks"][0][0, 0, 0] = float("nan")
            elif change == "dtype":
                after["generated_masks"][0] = after["generated_masks"][0].bfloat16()
            elif change == "shape":
                after["generated_masks"] = [torch.ones(1, 2, 3)]
            elif change == "missing_seg":
                before["generated_masks"] = after["generated_masks"] = [torch.empty(0, 2, 2)]
            else:
                before["generated_masks"] = after["generated_masks"] = [torch.ones(2, 2, 2)]
            with self.subTest(change=change), self.assertRaises(export.ProbeComparisonError):
                self.compare(before, after, policy="merge", skip_teacher_probe=True)

    def test_skip_keeps_strict_reload_gate_on_generated_mask_logits(self):
        before, after = self.generation_only_probe(), self.generation_only_probe()
        after["generated_masks"][0] += 0.25
        self.assertTrue(self.compare(before, after, policy="merge", skip_teacher_probe=True)["passed"])
        with self.assertRaisesRegex(export.ProbeComparisonError, "raw-logit"):
            self.compare(before, after, policy="strict", skip_teacher_probe=True)

    def test_raw_logit_failure_still_reports_identical_binary_masks_and_all_tensors(self):
        before = self.probe()
        after = deepcopy(before)
        after["generated_masks"][0][0, 0, 0] += 0.25
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "comparison.json"
            with self.assertRaises(export.ProbeComparisonError) as raised:
                self.compare(before, after, stage="unmerged_vs_merged", diagnostics_path=path)
            report = json.loads(path.read_text())
            self.assertEqual(report, raised.exception.report)
            self.assertFalse(report["passed"])
            self.assertTrue(report["answers_equal"])
            self.assertEqual(len(report["tensors"]), 3)
            entry = report["tensors"][0]
            self.assertEqual(entry["name"], "generated_masks[0]")
            self.assertEqual(entry["outside_tolerance_count"], 1)
            self.assertEqual(entry["outside_tolerance_fraction"], 0.25)
            self.assertEqual(entry["max_abs_error"], 0.25)
            self.assertEqual(entry["mean_abs_error"], 0.0625)
            self.assertEqual(entry["max_abs_error_index"], [0, 0, 0])
            self.assertEqual(entry["p50_abs_error"], 0)
            self.assertGreater(entry["p99_abs_error"], entry["p95_abs_error"])
            self.assertEqual(entry["binary_changed_pixels"], 0)
            self.assertEqual(entry["binary_mask_iou"], 1)
            self.assertFalse(entry["logits_close"])

    def test_sign_flip_is_not_hidden_by_passing_raw_logit_tolerance(self):
        before = self.probe()
        after = deepcopy(before)
        before["generated_masks"][0][0, 0, 0] = 0.0001
        after["generated_masks"][0][0, 0, 0] = -0.0001
        with self.assertRaisesRegex(export.ProbeComparisonError, "disagreement") as raised:
            self.compare(before, after)
        entry = raised.exception.report["tensors"][0]
        self.assertTrue(entry["logits_close"])
        self.assertEqual(entry["binary_changed_pixels"], 1)
        self.assertEqual(entry["binary_disagreement_fraction"], 0.25)
        self.assertEqual(entry["binary_mask_iou"], 0.75)
        self.assertEqual(entry["per_mask"][0]["binary_mask_iou"], 0.75)

    def test_shape_nonfinite_and_group_count_failures_are_serializable(self):
        for field, value in (("generated_masks", [torch.ones(1, 2, 3)]),
                             ("teacher_masks", [torch.full((1, 2, 2), float("nan"))]),
                             ("generated_masks", [])):
            with self.subTest(field=field, shape=[list(t.shape) for t in value]):
                before, after = self.probe(), self.probe()
                after[field] = value
                with self.assertRaises(export.ProbeComparisonError) as raised:
                    self.compare(before, after)
                report = raised.exception.report
                self.assertFalse(report["passed"])
                json.dumps(report, allow_nan=False)

    def test_answer_failure_still_reports_masks(self):
        before, after = self.probe(), self.probe()
        after["answers"] = ["A different answer."]
        with self.assertRaises(export.ProbeComparisonError) as raised:
            self.compare(before, after)
        report = raised.exception.report
        self.assertFalse(report["answers_equal"])
        self.assertEqual(len(report["tensors"]), 3)
        self.assertEqual(report["tensors"][0]["binary_mask_iou"], 1)

    def test_reference_nonfinite_values_and_dtype_changes_remain_failures(self):
        before, after = self.probe(), self.probe()
        before["logits"][0, 0, 0] = float("inf")
        with self.assertRaises(export.ProbeComparisonError) as raised:
            self.compare(before, after)
        self.assertEqual(raised.exception.report["tensors"][-1]["reference_nonfinite_count"], 1)
        before, after = self.probe(), self.probe()
        after["logits"] = after["logits"].bfloat16()
        with self.assertRaises(export.ProbeComparisonError) as raised:
            self.compare(before, after)
        self.assertIn("dtype", raised.exception.report["tensors"][-1]["assert_close_error"])

    def test_empty_masks_report_no_changed_pixels_without_nan(self):
        before, after = self.probe(), self.probe()
        before["generated_masks"] = [torch.empty(0, 2, 2)]
        after["generated_masks"] = [torch.empty(0, 2, 2)]
        report = self.compare(before, after)
        entry = report["tensors"][0]
        self.assertTrue(report["passed"])
        self.assertEqual(entry["element_count"], 0)
        self.assertEqual(entry["outside_tolerance_fraction"], 0)
        self.assertEqual(entry["binary_mask_iou"], 1)
        self.assertEqual(entry["per_mask"], [])
        json.dumps(report, allow_nan=False)

    def test_full_pipeline_failure_does_not_save_export_or_success_marker(self):
        before, after = self.probe(), self.probe()
        after["generated_masks"][0][0, 0, 0] = -1
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "image.jpg"
            image.write_bytes(b"existence-only mocked probe")
            model = SimpleNamespace()
            model.cuda = lambda: model
            model.eval = lambda: model
            model.parameters = lambda: iter([torch.ones(1, dtype=torch.bfloat16)])
            model.qwen = SimpleNamespace(merge_and_unload=lambda **kwargs: object())
            with patch.object(export, "restore_for_export", return_value=(model, {}, "epoch_020")), \
                 patch.object(export, "export_probe", side_effect=[before, after]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(export.ProbeComparisonError):
                    export.main(["--run_dir", str(root / "run"), "--output_dir", str(root / "out"),
                                 "--verify_image", str(image)])
            self.assertFalse((root / "out").exists())
            reports = list((root / "run" / "export_diagnostics").glob("*/unmerged_vs_merged.json"))
            self.assertEqual(len(reports), 1)
            self.assertFalse(json.loads(reports[0].read_text())["passed"])
            self.assertEqual(json.loads((reports[0].parent / "attempt.json").read_text())["source_tag"], "epoch_020")

    def test_success_reports_merge_and_reload_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "image.jpg"
            image.write_bytes(b"existence-only mocked probe")
            model = SimpleNamespace()
            model.cuda = lambda: model
            model.eval = lambda: model
            model.parameters = lambda: iter([torch.ones(1, dtype=torch.bfloat16)])
            model.qwen = SimpleNamespace(merge_and_unload=lambda **kwargs: object())
            model.save_pretrained = lambda path: path.mkdir()
            metadata = {"composite": {"source_qwen_revision": "a" * 40}}
            with patch.object(export, "restore_for_export", return_value=(model, metadata, "epoch_020")), \
                 patch.object(export, "export_probe", side_effect=[self.probe(), self.probe(), self.probe()]), \
                 patch.object(export.QwenSAM, "from_pretrained", return_value=model), \
                 patch.object(export.torch.cuda, "empty_cache"), contextlib.redirect_stdout(io.StringIO()):
                export.main(["--run_dir", str(root / "run"), "--output_dir", str(root / "out"),
                             "--verify_image", str(image)])
            verification = json.loads((root / "out" / "export_verification.json").read_text())
            self.assertTrue(verification["passed"])
            self.assertFalse(verification["teacher_probe_skipped"])
            for field, stage in (("merge_comparison", "unmerged_vs_merged"),
                                 ("serialization_comparison", "merged_vs_reloaded"),
                                 ("reload_comparison", "unmerged_vs_reloaded")):
                self.assertEqual(verification[field]["stage"], stage)
                self.assertTrue(Path(verification[field]["diagnostics_path"]).is_file())

    def test_merge_records_raw_logit_drift_but_strict_reload_rejects_it(self):
        before, after = self.probe(), self.probe()
        after["generated_masks"][0][0, 0, 0] += 0.25
        after["teacher_masks"][0] += 0.2
        after["logits"] += 0.5
        report = self.compare(before, after, policy="merge")
        self.assertTrue(report["passed"])
        self.assertTrue(report["diagnostic_warnings"])
        self.assertEqual(report["max_mask_disagreement"], 0)
        self.assertFalse(report["tensors"][0]["raw_logit_gate_applied"])
        self.assertEqual(report["tensors"][-1]["argmax_agreement"], 1)
        self.assertAlmostEqual(report["tensors"][-1]["mean_kl_reference_to_candidate"], 0)
        with self.assertRaises(export.ProbeComparisonError):
            self.compare(before, after, policy="strict")

    def test_merge_still_rejects_changed_answer_mask_and_nonfinite_or_dtype(self):
        for change in ("answer", "mask", "nonfinite", "dtype", "missing_seg", "multiple_seg"):
            before, after = self.probe(), self.probe()
            if change == "answer":
                after["answers"] = ["A different answer."]
            elif change == "mask":
                after["generated_masks"][0][0, 0, 0] = -1
            elif change == "nonfinite":
                after["logits"][0, 0, 0] = float("nan")
            elif change == "dtype":
                after["logits"] = after["logits"].bfloat16()
            elif change == "missing_seg":
                before["generated_masks"] = after["generated_masks"] = [torch.empty(0, 2, 2)]
            else:
                before["generated_masks"] = after["generated_masks"] = [torch.ones(2, 2, 2)]
            with self.subTest(change=change), self.assertRaises(export.ProbeComparisonError):
                self.compare(before, after, policy="merge")

    def test_subset_reports_all_images_even_when_first_image_fails(self):
        before = [self.probe(), self.probe()]
        after = [self.probe(), self.probe()]
        after[0]["answers"] = ["changed"]
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            with self.assertRaises(export.ProbeComparisonError) as raised:
                export.compare_probe_sets(before, after, ["/val/a.jpg", "/val/b.jpg"], 0.03, 0.001,
                    stage="unmerged_vs_merged", policy="merge", diagnostics=root)
            report = json.loads((root / "unmerged_vs_merged.json").read_text())
            self.assertEqual(report, raised.exception.report)
            self.assertEqual(report["sample_count"], 2)
            self.assertEqual(report["passed_sample_count"], 1)
            self.assertFalse(report["results"][0]["passed"])
            self.assertTrue(report["results"][1]["passed"])
            self.assertTrue((root / "unmerged_vs_merged" / "01_b.json").is_file())

    def test_manifest_preserves_order_and_requires_all_selected_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.jpg").write_bytes(b"a")
            (root / "b.jpg").write_bytes(b"b")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "image_dir": ".", "images": ["b.jpg", "a.jpg"]}))
            self.assertEqual(export.load_verification_images(manifest=manifest),
                             [str((root / "b.jpg").resolve()), str((root / "a.jpg").resolve())])
            for names, error in ((["a.jpg", "a.jpg"], ValueError), (["a.jpg", "missing.jpg"], FileNotFoundError)):
                manifest.write_text(json.dumps({"schema_version": 1, "image_dir": ".", "images": names}))
                with self.assertRaises(error):
                    export.load_verification_images(manifest=manifest)
            with self.assertRaises(ValueError):
                export.load_verification_images(image=root / "a.jpg", manifest=manifest)

    def test_default_manifest_is_fixed_eight_validation_not_test_images(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "qwen_export_validation.json"
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["image_dir"], "/workspace/datasets/demo_dataset/reason_seg/ReasonSeg/val")
        self.assertEqual(manifest["images"], ["rs08006.jpg", "rs08016.jpg", "rs08051.jpg", "rs08071.jpg",
                                              "rs08299.jpg", "rs08302.jpg", "rs08306.jpg", "rs08319.jpg"])

    def test_pipeline_accepts_merge_drift_but_never_accepts_reload_drift(self):
        for reload_drift in (False, True):
            with self.subTest(reload_drift=reload_drift), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                image = root / "image.jpg"
                image.write_bytes(b"mocked probe")
                before, merged = self.probe(), self.probe()
                merged["generated_masks"][0] += 0.25
                merged["logits"] += 0.5
                reloaded = deepcopy(merged)
                if reload_drift:
                    reloaded["teacher_masks"][0] += 0.3
                model = SimpleNamespace()
                model.cuda = lambda: model
                model.eval = lambda: model
                model.parameters = lambda: iter([torch.ones(1, dtype=torch.bfloat16)])
                model.qwen = SimpleNamespace(merge_and_unload=lambda **kwargs: object())
                model.save_pretrained = lambda path: path.mkdir()
                metadata = {"composite": {"source_qwen_revision": "a" * 40}}
                with patch.object(export, "restore_for_export", return_value=(model, metadata, "epoch_020")), \
                     patch.object(export, "export_probe", side_effect=[before, merged, reloaded]), \
                     patch.object(export.QwenSAM, "from_pretrained", return_value=model), \
                     patch.object(export.torch.cuda, "empty_cache"), contextlib.redirect_stdout(io.StringIO()):
                    arguments = ["--run_dir", str(root / "run"), "--output_dir", str(root / "out"),
                                 "--verify_image", str(image)]
                    if reload_drift:
                        with self.assertRaises(export.ProbeComparisonError):
                            export.main(arguments)
                    else:
                        export.main(arguments)
                marker = root / "out" / "export_verification.json"
                self.assertEqual(marker.exists(), not reload_drift)
                if not reload_drift:
                    report = json.loads(marker.read_text())
                    self.assertEqual(report["merge_comparison"]["policy"], "merge")
                    self.assertEqual(report["serialization_comparison"]["policy"], "strict")

    def test_manifest_pipeline_probes_every_image_at_all_three_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = [root / "b.jpg", root / "a.jpg"]
            for image in images:
                image.write_bytes(b"existence-only mocked probe")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "image_dir": ".",
                                            "images": [image.name for image in images]}))
            model = SimpleNamespace()
            model.cuda = lambda: model
            model.eval = lambda: model
            model.parameters = lambda: iter([torch.ones(1, dtype=torch.bfloat16)])
            model.qwen = SimpleNamespace(merge_and_unload=lambda **kwargs: object())
            model.save_pretrained = lambda path: path.mkdir()
            metadata = {"composite": {"source_qwen_revision": "a" * 40}}
            with patch.object(export, "restore_for_export", return_value=(model, metadata, "epoch_020")), \
                 patch.object(export, "export_probe", side_effect=[self.probe() for _ in range(6)]) as probe, \
                 patch.object(export.QwenSAM, "from_pretrained", return_value=model), \
                 patch.object(export.torch.cuda, "empty_cache"), contextlib.redirect_stdout(io.StringIO()):
                export.main(["--run_dir", str(root / "run"), "--output_dir", str(root / "out"),
                             "--verify_manifest", str(manifest)])
            ordered = [str(image.resolve()) for image in images]
            self.assertEqual([call.args[1] for call in probe.call_args_list], ordered * 3)
            self.assertTrue(all(call.kwargs == {"skip_teacher_probe": False} for call in probe.call_args_list))
            verification = json.loads((root / "out" / "export_verification.json").read_text())
            self.assertEqual(verification["verification_images"], ordered)
            for field in ("merge_comparison", "serialization_comparison", "reload_comparison"):
                self.assertEqual(verification[field]["sample_count"], 2)
                self.assertEqual(verification[field]["passed_sample_count"], 2)
                self.assertEqual([report["image"] for report in verification[field]["results"]], ordered)

    def test_skip_pipeline_preserves_generation_and_reload_verification(self):
        for failure in (None, "merge_answer", "merge_mask", "reload_mask_logits"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                images = [root / "a.jpg", root / "b.jpg"]
                for image in images:
                    image.write_bytes(b"existence-only mocked probe")
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({"schema_version": 1, "image_dir": ".",
                                                "images": [image.name for image in images]}))
                probes = [self.generation_only_probe() for _ in range(6)]
                if failure == "merge_answer":
                    probes[2]["answers"] = ["Changed answer."]
                elif failure == "merge_mask":
                    probes[2]["generated_masks"][0][0, 0, 0] = -1
                elif failure == "reload_mask_logits":
                    probes[4]["generated_masks"][0] += 0.25
                model = SimpleNamespace()
                model.cuda = lambda: model
                model.eval = lambda: model
                model.parameters = lambda: iter([torch.ones(1, dtype=torch.bfloat16)])
                model.qwen = SimpleNamespace(merge_and_unload=lambda **kwargs: object())
                model.save_pretrained = lambda path: path.mkdir()
                metadata = {"composite": {"source_qwen_revision": "a" * 40}}
                with patch.object(export, "restore_for_export", return_value=(model, metadata, "epoch_020")), \
                     patch.object(export, "export_probe", side_effect=probes) as probe, \
                     patch.object(export.QwenSAM, "from_pretrained", return_value=model), \
                     patch.object(export.torch.cuda, "empty_cache"), contextlib.redirect_stdout(io.StringIO()):
                    arguments = ["--run_dir", str(root / "run"), "--output_dir", str(root / "out"),
                                 "--verify_manifest", str(manifest), "--skip_teacher_probe"]
                    if failure is None:
                        export.main(arguments)
                    else:
                        with self.assertRaises(export.ProbeComparisonError):
                            export.main(arguments)
                self.assertTrue(all(call.kwargs == {"skip_teacher_probe": True} for call in probe.call_args_list))
                self.assertEqual(probe.call_count, 4 if failure in ("merge_answer", "merge_mask") else 6)
                attempt_path, = (root / "run" / "export_diagnostics").glob("*/attempt.json")
                self.assertTrue(json.loads(attempt_path.read_text())["teacher_probe_skipped"])
                marker = root / "out" / "export_verification.json"
                self.assertEqual(marker.exists(), failure is None)
                if failure in ("merge_answer", "merge_mask"):
                    self.assertFalse((root / "out").exists())
                if failure is None:
                    report = json.loads(marker.read_text())
                    self.assertTrue(report["teacher_probe_skipped"])
                    ordered = [str(image.resolve()) for image in images]
                    self.assertEqual([call.args[1] for call in probe.call_args_list], ordered * 3)
                    for field in ("merge_comparison", "serialization_comparison", "reload_comparison"):
                        self.assertTrue(report[field]["teacher_probe_skipped"])
                        self.assertTrue(report[field]["passed"])
                        for sample in report[field]["results"]:
                            self.assertTrue(sample["teacher_probe_skipped"])
                            self.assertEqual([entry["name"] for entry in sample["tensors"]], ["generated_masks[0]"])


if __name__ == "__main__":
    unittest.main()
