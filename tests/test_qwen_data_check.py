"""Stdlib safety/reporting checks, independent of the native processing gate."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qwen_data_check as check


class DataCheckControlsTest(unittest.TestCase):
    def argv(self, temp, **changes):
        paths = {"model-dir": str(Path(temp) / "model"),
                 "dataset-root": str(Path(temp) / "datasets"),
                 "report": str(Path(temp) / "check.json")}
        paths.update(changes)
        return [item for key, value in paths.items() for item in ("--" + key, value)]

    def test_default_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            args = check.parse_args(self.argv(temp))
        self.assertEqual((args.min_pixels, args.max_pixels, args.max_seq_length),
                         (16384, 1048576, 4096))

    def test_source_directories_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            for directory in ("model", "datasets"):
                with self.subTest(directory=directory), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        check.parse_args(self.argv(temp, report=str(Path(temp) / directory / "check.json")))

    def test_existing_reports_and_processor_outputs_preserved(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            report = Path(temp) / "previous.json"
            report.write_text("previous result")
            with self.assertRaises(SystemExit):
                check.parse_args(self.argv(temp, report=str(report)))
            self.assertEqual(report.read_text(), "previous result")
            (Path(temp) / "check_processor").mkdir()
            with self.assertRaises(SystemExit):
                check.parse_args(self.argv(temp))

    def test_invalid_budget_rejected(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            for changes in ({"samples-per-source": "0"}, {"max-pixels": "1"},
                            {"max-seq-length": "0"}):
                with self.subTest(changes=changes), self.assertRaises(SystemExit):
                    check.parse_args(self.argv(temp, **changes))

    def test_failed_gate_reports_error_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(check, "run_check", side_effect=ValueError("bad token alignment")), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(check.main(self.argv(temp)), 1)
            report = json.loads((Path(temp) / "check.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"], "bad token alignment")

    def test_success_reporting_not_native_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(check, "run_check"), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check.main(self.argv(temp)), 0)
            self.assertEqual(json.loads((Path(temp) / "check.json").read_text())["status"], "passed")

    def test_launcher_binds_sources_read_only_and_keeps_known_qos(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "qwen_data_check.sbatch").read_text()
        self.assertIn("#SBATCH --qos=gpu32-1h", launcher)
        self.assertIn('$QWEN_DATA:/workspace/datasets:ro', launcher)
        self.assertIn('$QWEN_MODEL:/workspace/qwen_model:ro', launcher)
        self.assertIn("/scratch/mohammadjavad/Qwen3VL-v1.sif", launcher)


if __name__ == "__main__":
    unittest.main()
