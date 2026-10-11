"""CPU/stdlib checks for Stage 1 controls; not a substitute for the CUDA gate."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import qwen_environment as env


def manifest(status="ready", sha="a" * 40):
    return {"schema_version": 1, "model_id": env.MODEL_ID,
            "resolved_revision": sha, "status": status}


class EnvironmentControlsTest(unittest.TestCase):
    def test_training_launcher_sam_defaults_overrides_and_resume(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "fine_tune_Qwen_2nodes.sbatch").read_text()
        declarations = [line for line in launcher.splitlines()
                        if line.startswith(("export QWEN_SAM_INIT_SOURCE=", "export QWEN_SAM_INIT_PATH="))]
        self.assertEqual(declarations, [
            'export QWEN_SAM_INIT_SOURCE="${QWEN_SAM_INIT_SOURCE:-pretrained}"',
            'export QWEN_SAM_INIT_PATH="${QWEN_SAM_INIT_PATH:-/workspace/checkpoints/sam_vit_h_4b8939.pth}"'])
        branch = "initialization=()\n" + launcher.split("initialization=()\n", 1)[1].split("run_dir=", 1)[0]
        command = "\n".join(declarations) + "\n" + branch + '\nprintf "%s\\n" "${initialization[@]}"'
        cases = [({}, ["--model_dir=/workspace/qwen_model", "--sam_init_source=pretrained",
                       "--sam_init_path=/workspace/checkpoints/sam_vit_h_4b8939.pth"]),
                 ({"QWEN_SAM_INIT_SOURCE": "rail_checkpoint", "QWEN_SAM_INIT_PATH": "/custom/rail.pt"},
                  ["--model_dir=/workspace/qwen_model", "--sam_init_source=rail_checkpoint",
                   "--sam_init_path=/custom/rail.pt"]),
                 ({"QWEN_RESUME": "1", "QWEN_RUN_DIR": "/custom/run"}, ["--resume"])]
        for runtime_env, expected in cases:
            with self.subTest(environment=runtime_env):
                result = subprocess.run(["/bin/bash", "-eu", "-c", command], env=runtime_env,
                                        capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout.splitlines(), expected)

    def test_all_qwen_launchers_default_to_v1_and_allow_overrides(self):
        root = Path(__file__).resolve().parents[1]
        default = "/scratch/mohammadjavad/Qwen3VL-v1.sif"
        for filename in ("fine_tune_Qwen_2nodes.sbatch", "merge_Qwen.sbatch", "demo_Qwen.sbatch",
                         "qwen_data_check.sbatch", "qwen_environment.sbatch", "qwen_hpc_run.sh"):
            with self.subTest(filename=filename):
                declarations = [line for line in (root / filename).read_text().splitlines()
                                if line.startswith("export QWEN_IMG=")]
                self.assertEqual(declarations, ['export QWEN_IMG="${QWEN_IMG:-' + default + '}"'])
                for runtime_env, expected in (({}, default), ({"QWEN_IMG": "/custom/qwen.sif"}, "/custom/qwen.sif")):
                    result = subprocess.run(["/bin/bash", "-c", declarations[0] + '\nprintf "%s" "$QWEN_IMG"'],
                                            env=runtime_env, capture_output=True, text=True, check=True)
                    self.assertEqual(result.stdout, expected)

    def test_docker_requirements_and_runtime_stack_agree(self):
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "containers/qwen/Dockerfile").read_text()
        requirements = (root / "containers/qwen/requirements.txt").read_text()
        launcher = (root / "qwen_environment.sbatch").read_text()
        for package, version in env.REQUIRED_VERSIONS.items():
            pinned = version + "+cu118" if package in ("torch", "torchvision") else version
            self.assertIn(f"{package}=={pinned}", requirements.splitlines())
            self.assertIn(f'{package}.__version__ == "{pinned}"', dockerfile)
        self.assertIn("FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel@sha256:", dockerfile)
        self.assertIn("/opt/conda/bin/python /workspace/home/qwen_environment.py", launcher)
        self.assertNotIn("/opt/qwen/venv", launcher)

    def test_versions_accept_cuda_suffix_but_not_wrong_stack(self):
        versions = dict(env.REQUIRED_VERSIONS)
        versions["torch"] += "+cu118"
        versions["torchvision"] += "+cu118"
        self.assertEqual(env.version_errors(versions), [])
        versions["transformers"] = "4.31.0"
        self.assertEqual(len(env.version_errors(versions)), 1)

    def test_missing_packages_fail(self):
        self.assertEqual(len(env.version_errors({})), 5)

    def test_python_is_recorded_without_an_exact_version_gate(self):
        report = {}
        args = SimpleNamespace(model_dir=Path("missing-model"))
        with patch.object(env, "package_versions", return_value=dict(env.REQUIRED_VERSIONS)):
            with patch.object(env, "load_manifest", side_effect=ValueError("missing manifest")):
                with self.assertRaisesRegex(ValueError, "missing manifest"):
                    env.run_smoke(args, report)
        self.assertEqual(report["python"], env.sys.version)

    def test_manifest_requires_model_sha_and_completion(self):
        self.assertEqual(env.validate_manifest(manifest())["resolved_revision"], "a" * 40)
        for changed in ({"model_id": "other"}, {"schema_version": 2},
                        {"resolved_revision": "main"}, {"status": "downloading"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                env.validate_manifest({**manifest(), **changed})
        env.validate_manifest(manifest("downloading"), require_ready=False)

    def test_missing_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            env.load_manifest(temp)

    def test_budget_checks_total_length_without_truncation(self):
        env.check_sequence_budget(3968, 128, 4096)
        for count in (0, 3969):
            with self.subTest(count=count), self.assertRaises(ValueError):
                env.check_sequence_budget(count, 128, 4096)

    def smoke_argv(self, temp, **changes):
        paths = {"model-dir": str(Path(temp) / "model"),
                 "image": str(Path(temp) / "image.jpg"),
                 "report": str(Path(temp) / "report.json")}
        paths.update(changes)
        return ["smoke"] + [item for key, value in paths.items()
                            for item in ("--" + key, value)]

    def test_smoke_defaults_match_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            args = env.parse_args(self.smoke_argv(temp))
        self.assertEqual((args.min_pixels, args.max_pixels, args.max_seq_length),
                         (16384, 1048576, 4096))
        self.assertEqual(args.prompt, env.DEFAULT_PROMPT)

    def test_invalid_bounds_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for changes in ({"max-new-tokens": "0"},
                            {"min-pixels": "999", "max-pixels": "1"},
                            {"max-new-tokens": "4096"}):
                with self.subTest(changes=changes), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        env.parse_args(self.smoke_argv(temp, **changes))

    def test_report_cannot_overwrite_image_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            for report in (Path(temp) / "image.jpg", Path(temp) / "model/config.json"):
                with self.subTest(report=report), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        env.parse_args(self.smoke_argv(temp, report=str(report)))

    def test_failure_saves_report_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            args = env.parse_args(self.smoke_argv(temp))
            with patch.object(env, "run_smoke", side_effect=RuntimeError("CUDA unavailable")):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(env.smoke(args), 1)
            report = json.loads(args.report.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("CUDA unavailable", report["error"])
            self.assertNotIn("generation_passed", report)

    def test_success_report_control_flow(self):
        # Only tests reporting around the GPU function, not model execution.
        with tempfile.TemporaryDirectory() as temp:
            args = env.parse_args(self.smoke_argv(temp))
            with patch.object(env, "run_smoke"), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(env.smoke(args), 0)
            self.assertEqual(json.loads(args.report.read_text())["status"], "passed")


class SnapshotDownloadTest(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.api.model_info.return_value = SimpleNamespace(sha="a" * 40)
        self.hub = SimpleNamespace(HfApi=Mock(return_value=self.api), snapshot_download=Mock())
        self.modules = patch.dict("sys.modules", {"huggingface_hub": self.hub})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_download_pins_sha_and_marks_ready(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            args = env.parse_args(["download", "--output-dir", str(Path(temp) / "model")])
            env.download_snapshot(args)
            self.assertEqual(self.hub.snapshot_download.call_args.kwargs["revision"], "a" * 40)
            self.assertEqual(env.load_manifest(args.output_dir)["status"], "ready")

    def test_failed_download_is_incomplete_and_same_sha_retryable(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            args = env.parse_args(["download", "--output-dir", str(Path(temp) / "model")])
            self.hub.snapshot_download.side_effect = RuntimeError("network interrupted")
            with self.assertRaises(RuntimeError):
                env.download_snapshot(args)
            with self.assertRaises(ValueError):
                env.load_manifest(args.output_dir)
            self.hub.snapshot_download.side_effect = None
            env.download_snapshot(args)
            self.assertEqual(env.load_manifest(args.output_dir)["status"], "ready")

    def test_refuses_nonempty_unmanaged_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)
            file = destination / "config.json"
            file.write_text("existing")
            args = env.parse_args(["download", "--output-dir", temp])
            with self.assertRaises(ValueError):
                env.download_snapshot(args)
            self.assertEqual(file.read_text(), "existing")
            self.hub.snapshot_download.assert_not_called()

    def test_refuses_revision_change(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)
            env.write_json(destination / env.MANIFEST_NAME, manifest(sha="b" * 40))
            args = env.parse_args(["download", "--output-dir", temp])
            with self.assertRaises(ValueError):
                env.download_snapshot(args)
            self.assertEqual(env.load_manifest(destination)["resolved_revision"], "b" * 40)
            self.hub.snapshot_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
