"""Launch defaults and LISA-compatible output checks; no GPU/checkpoint needed."""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class QwenInferenceLauncherTest(unittest.TestCase):
    def launch(self, filename, arguments=(), environment=None):
        # Exercise the actual Bash argument construction without Slurm or HPC.
        command = 'srun() { printf "%s\\n" "$@"; }\n' + (ROOT / filename).read_text()
        result = subprocess.run(["/bin/bash", "-eu", "-c", command, filename, *arguments],
                                env=environment or {}, capture_output=True, text=True, check=True)
        return result.stdout.splitlines()

    def test_merge_and_demo_have_matching_self_contained_defaults(self):
        merge = self.launch("merge_Qwen.sbatch")
        demo = self.launch("demo_Qwen.sbatch")
        for arguments in (merge, demo):
            self.assertIn("/scratch/mohammadjavad/QWEN-LISA_Codebase/qwen_hpc_run.sh", arguments)
        self.assertIn("--run_dir=/workspace/home/runs/qwen_159894", merge)
        self.assertIn("--output_dir=/workspace/home/exports/qwen_159894", merge)
        self.assertIn("--max_new_tokens=512", merge)
        self.assertIn("--skip_teacher_probe", merge)
        self.assertIn("--verify_manifest=/workspace/home/configs/qwen_export_validation.json", merge)
        self.assertFalse(any(arg.startswith("--tag") for arg in merge))
        self.assertIn("--model_path=/workspace/home/exports/qwen_159894", demo)
        self.assertIn("--image_path=/workspace/checkpoints/rs19_crop_1024", demo)
        self.assertIn("--output_dir=/workspace/home/test_outputs/qwen_159894", demo)
        self.assertIn("--mask_save_path=/workspace/home/mask_output", demo)
        self.assertIn("--vis_save_path=/workspace/home/vis_output", demo)
        self.assertIn("--max_new_tokens=512", demo)
        self.assertNotIn("--evaluate_masks", demo)
        self.assertIn("--prompt=Based on the blade positions in this switch, which route corresponds to "
                      "the route the train takes? Please respond with segmentation mask and explain why.", demo)

    def test_merge_paths_and_epoch_can_be_overridden(self):
        arguments = self.launch("merge_Qwen.sbatch", environment={
            "QWEN_CODE": "/custom/code", "QWEN_RUN_DIR": "/custom/run",
            "QWEN_EXPORT_DIR": "/custom/export", "QWEN_VERIFY_IMAGE": "/custom/image.jpg",
            "QWEN_CHECKPOINT_TAG": "epoch_003"})
        for argument in ("/custom/code/qwen_hpc_run.sh", "--run_dir=/custom/run",
                         "--output_dir=/custom/export", "--verify_image=/custom/image.jpg", "--tag=epoch_003"):
            self.assertIn(argument, arguments)

    def test_explicit_inference_source_overrides_default_without_conflicting_sources(self):
        for explicit in (("--manifest=/custom/labels.json",), ("--manifest", "/custom/labels.json"),
                         ("--image_path=/custom/images",), ("--image_path", "/custom/images")):
            with self.subTest(arguments=explicit):
                arguments = self.launch("demo_Qwen.sbatch", explicit)
                self.assertNotIn("--image_path=/workspace/checkpoints/rs19_crop_1024", arguments)
                for argument in explicit:
                    self.assertIn(argument, arguments)

    def test_explicit_merge_verification_source_overrides_manifest(self):
        for explicit in (("--verify_image=/custom/image.jpg",), ("--verify_image", "/custom/image.jpg"),
                         ("--verify_manifest=/custom/images.json",)):
            with self.subTest(arguments=explicit):
                arguments = self.launch("merge_Qwen.sbatch", explicit)
                self.assertNotIn("--verify_manifest=/workspace/home/configs/qwen_export_validation.json", arguments)
                for argument in explicit:
                    self.assertIn(argument, arguments)

    def test_inference_defaults_allow_environment_overrides(self):
        arguments = self.launch("demo_Qwen.sbatch", environment={
            "QWEN_EXPORT_DIR": "/custom/export", "QWEN_RESULTS_DIR": "/custom/results",
            "QWEN_TEST_IMAGES": "/custom/images", "QWEN_MASK_SAVE_PATH": "/custom/masks",
            "QWEN_VIS_SAVE_PATH": "/custom/overlays", "QWEN_PROMPT": "A custom prompt",
            "QWEN_MAX_NEW_TOKENS": "128"})
        for argument in ("--model_path=/custom/export", "--output_dir=/custom/results",
                         "--image_path=/custom/images", "--mask_save_path=/custom/masks",
                         "--vis_save_path=/custom/overlays", "--prompt=A custom prompt", "--max_new_tokens=128"):
            self.assertIn(argument, arguments)

    def test_launchers_keep_longer_time_and_known_node_exclusions(self):
        for name in ("merge_Qwen.sbatch", "demo_Qwen.sbatch"):
            source = (ROOT / name).read_text()
            self.assertIn("#SBATCH -t 04:00:00", source)
            self.assertIn("#SBATCH --gres=gpu:1", source)
            self.assertIn("#SBATCH --exclude=lse-hpcnode5,lse-hpcnode7,lse-hpcnode14", source)
        self.assertIn("#SBATCH --mem=192G", (ROOT / "merge_Qwen.sbatch").read_text())


ML_ERROR = None
try:
    import cv2
    import numpy as np
    import torch
    import chat_batch_qwen as inference
    from utils.qwen_evaluation import save_lisa_compatible_outputs
except ModuleNotFoundError as exc:
    ML_ERROR = str(exc)


@unittest.skipIf(ML_ERROR, "Qwen output dependencies unavailable: " + str(ML_ERROR))
class QwenLegacyOutputTest(unittest.TestCase):
    def test_jpegs_match_original_lisa_encoding_and_overlay(self):
        rgb = np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
        mask = torch.zeros(16, 20, dtype=torch.bfloat16)
        mask[3:12, 4:15] = 1
        foreground = mask.float().numpy() > 0
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mask_path, vis_path = root / "rs06001.jpg", root / "overlay.jpg"
            save_lisa_compatible_outputs(mask, rgb, mask_path, vis_path)
            expected_mask = foreground.astype(np.uint8) * 100
            expected_overlay = rgb.copy()
            expected_overlay[foreground] = (
                rgb * 0.5 + foreground[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[foreground]
            self.assertEqual(mask_path.read_bytes(), cv2.imencode(".jpg", expected_mask)[1].tobytes())
            self.assertEqual(vis_path.read_bytes(), cv2.imencode(
                ".jpg", cv2.cvtColor(expected_overlay, cv2.COLOR_RGB2BGR))[1].tobytes())

    def test_nonfinite_masks_and_image_size_mismatch_rejected(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaises(FloatingPointError):
            save_lisa_compatible_outputs(torch.full((2, 2), float("nan")), rgb)
        with self.assertRaises(ValueError):
            save_lisa_compatible_outputs(torch.ones(1, 2), rgb)

    def test_output_paths_protect_sources_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            requests = [{"image_path": str(root / "images" / "rs06001.jpg")}]
            arguments = (root / "results", root / "model", requests)
            self.assertEqual(len(inference.validate_output_directories(
                *arguments, root / "masks", root / "vis")), 3)
            for unsafe in (root / "model", root / "model" / "masks", root / "images", root):
                with self.subTest(path=unsafe), self.assertRaises(ValueError):
                    inference.validate_output_directories(*arguments, unsafe)
            (root / "masks").mkdir()
            (root / "masks" / "existing.jpg").write_bytes(b"preserve")
            with self.assertRaises(ValueError):
                inference.validate_output_directories(*arguments, root / "masks")
            self.assertEqual((root / "masks" / "existing.jpg").read_bytes(), b"preserve")
            with self.assertRaises(ValueError):
                inference.validate_output_directories(*arguments, root / "results")

    def test_legacy_filenames_cannot_collide(self):
        requests = [{"image_path": "/inputs/rs06001.jpg"}, {"image_path": "/inputs/rs06001.png"}]
        with self.assertRaisesRegex(ValueError, "stems"):
            inference.validate_output_directories("/results", "/model", requests, "/masks")

    def test_inference_writes_pngs_legacy_outputs_and_answer_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images, model_path = root / "images", root / "model"
            images.mkdir()
            model_path.mkdir()
            (model_path / "export_verification.json").write_text(json.dumps({"passed": True}))
            cv2.imwrite(str(images / "rs06001.jpg"), np.full((16, 20, 3), 80, dtype=np.uint8))
            model = SimpleNamespace(processor=None, processing_config=None)
            model.cuda = lambda: model
            model.eval = lambda: model
            model.generate_masks = lambda *args, **kwargs: {
                "answers": ["[SEG]. This is a turnout switch. The train takes the left route."],
                "pred_masks": [torch.ones(1, 16, 20, dtype=torch.bfloat16)]}
            with patch.object(inference.QwenSAM, "from_pretrained", return_value=model), \
                 patch.object(inference, "QwenSegCollator", return_value=lambda samples: {}), \
                 patch.object(inference, "move_batch", side_effect=lambda batch, device: batch):
                inference.main(["--model_path", str(model_path), "--output_dir", str(root / "results"),
                                "--image_path", str(images), "--mask_save_path", str(root / "masks"),
                                "--vis_save_path", str(root / "vis")])
            record = json.loads((root / "results" / "results.jsonl").read_text())
            self.assertNotIn("error", record)
            self.assertEqual(record["mask_count"], 1)
            self.assertEqual(record["legacy_mask_index"], 0)
            self.assertTrue((root / "results" / record["mask_files"][0]).is_file())
            self.assertTrue((root / "masks" / "rs06001.jpg").is_file())
            self.assertTrue((root / "vis" / "rs06001.jpg").is_file())
            self.assertEqual(json.loads((root / "results" / "metrics.json").read_text())["one_mask_rate"], 1)


if __name__ == "__main__":
    unittest.main()
