"""Prepare a revision-pinned Qwen snapshot and verify it on one CUDA GPU.

No LISA, CLIP, SAM, dataset mutation, training, or distributed initialization.
Heavy dependencies are deliberately imported only within the selected command.
"""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MANIFEST_NAME = "qwen_snapshot_manifest.json"
DEFAULT_PROMPT = (
    "Based on the blade positions in this switch, which route corresponds to "
    "the route the train takes? Please respond with segmentation mask and "
    "explain why."
)
REQUIRED_VERSIONS = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "transformers": "4.57.6",
    "peft": "0.17.1",
    "deepspeed": "0.17.6",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def validate_manifest(payload, require_ready=True):
    if payload.get("schema_version") != 1 or payload.get("model_id") != MODEL_ID:
        raise ValueError("Snapshot manifest must describe Qwen3-VL-8B-Instruct (schema 1).")
    if not re.fullmatch(r"[0-9a-f]{40}", str(payload.get("resolved_revision", ""))):
        raise ValueError("Snapshot manifest needs the immutable 40-character HF commit SHA.")
    if require_ready and payload.get("status") != "ready":
        raise ValueError("Snapshot download is incomplete; rerun the download command first.")
    return payload


def load_manifest(model_dir):
    path = Path(model_dir) / MANIFEST_NAME
    if not path.is_file():
        raise ValueError(
            f"Missing {path}. Prepare the checkpoint with this tool's download command "
            "so the actual HF revision is recorded."
        )
    return validate_manifest(json.loads(path.read_text()))


def version_errors(versions):
    errors = []
    for package, expected in REQUIRED_VERSIONS.items():
        actual = versions.get(package, "not installed")
        if actual.split("+", 1)[0] != expected:
            errors.append(f"Expected {package} {expected}, got {actual}.")
    return errors


def check_sequence_budget(input_tokens, max_new_tokens, max_seq_length):
    if input_tokens <= 0:
        raise ValueError("Processor returned an empty input sequence.")
    if input_tokens + max_new_tokens > max_seq_length:
        raise ValueError(
            f"Input ({input_tokens}) + generation budget ({max_new_tokens}) exceeds "
            f"the sequence limit ({max_seq_length}); no tokens were truncated."
        )


def download_snapshot(args):
    from huggingface_hub import HfApi, snapshot_download

    destination = args.output_dir.expanduser().resolve()
    manifest_path = destination / MANIFEST_NAME
    sha = HfApi().model_info(MODEL_ID, revision=args.revision).sha
    manifest = validate_manifest({
        "schema_version": 1,
        "model_id": MODEL_ID,
        "requested_revision": args.revision,
        "resolved_revision": sha,
        "status": "downloading",
        "started_at": utc_now(),
    }, require_ready=False)
    if manifest_path.is_file():
        previous = validate_manifest(json.loads(manifest_path.read_text()), require_ready=False)
        if previous["resolved_revision"] != sha:
            raise ValueError(
                "Destination contains a different revision. Use a new output directory "
                "or request the recorded SHA; existing model files were not overwritten."
            )
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty model directory without a manifest.")
    write_json(manifest_path, manifest)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=sha,
        local_dir=str(destination),
        # Native Transformers only; no remote Python code or duplicate weight formats.
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
        max_workers=4,
    )
    manifest.update(status="ready", completed_at=utc_now())
    write_json(manifest_path, manifest)
    print(json.dumps({"model_dir": str(destination), **manifest}, indent=2))


def package_versions():
    return dict(sorted((dist.metadata["Name"].lower(), dist.version)
                       for dist in metadata.distributions() if dist.metadata["Name"]))


def run_smoke(args, report):
    report["packages"] = package_versions()
    report["python"] = sys.version
    report["platform"] = platform.platform()
    # Use the official Docker image's Python; record it, but do not force 3.10.
    errors = version_errors(report["packages"])
    if errors:
        raise RuntimeError("Wrong environment: " + " ".join(errors))
    report["snapshot"] = load_manifest(args.model_dir)
    if not args.image.is_file():
        raise FileNotFoundError(f"Smoke image does not exist: {args.image}")

    import torch
    import torchvision  # noqa: F401 -- verify native operator/import compatibility
    import peft  # noqa: F401
    import deepspeed  # noqa: F401
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if torch.version.cuda != "11.8":
        raise RuntimeError(f"Expected CUDA 11.8 PyTorch wheels, got {torch.version.cuda}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this gate must run on a healthy NVIDIA GPU.")
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The assigned GPU does not support BF16.")
    device = torch.device("cuda:0")
    report["gpu"] = {
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_wheel_version": torch.version.cuda,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        report["gpu"]["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report["gpu"]["nvidia_smi_error"] = str(exc)
    torch.manual_seed(0)
    probe = torch.randn(32, 32, device=device, dtype=torch.bfloat16)
    if not torch.isfinite(probe @ probe.T).all().item():
        raise RuntimeError("CUDA BF16 matrix multiplication returned non-finite values.")
    torch.cuda.synchronize()
    del probe
    report["cuda_kernel_passed"] = True
    torch.cuda.reset_peak_memory_stats()

    processor = AutoProcessor.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    report["image_size_wh"] = list(image.size)
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": args.prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=False,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        if key not in inputs:
            raise RuntimeError(f"Native processor did not return {key}.")
    input_tokens = inputs["input_ids"].shape[-1]
    check_sequence_budget(input_tokens, args.max_new_tokens, args.max_seq_length)
    report["processor"] = {
        "class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "input_tokens": input_tokens,
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_seq_length": args.max_seq_length,
    }

    model, loading_info = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0},
        output_loading_info=True,
    )
    report["loading_info"] = loading_info
    if any(loading_info.get(key) for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
    )):
        raise RuntimeError("Native Qwen checkpoint did not load completely; see loading_info.")
    model.eval()
    grid = inputs["image_grid_thw"]
    merge = model.config.vision_config.spatial_merge_size
    expected_visual_tokens = int((grid.prod(dim=-1) // (merge * merge)).sum().item())
    actual_visual_tokens = int((inputs["input_ids"] == model.config.image_token_id).sum().item())
    report["processor"].update(
        expected_visual_tokens=expected_visual_tokens, actual_visual_tokens=actual_visual_tokens,
        resized_image_sizes_hw=[
            [int(row[1]) * model.config.vision_config.patch_size,
             int(row[2]) * model.config.vision_config.patch_size]
            for row in grid.tolist()
        ],
    )
    if actual_visual_tokens <= 0 or actual_visual_tokens != expected_visual_tokens:
        raise RuntimeError("Visual placeholders do not match Qwen's native image grid.")
    inputs = inputs.to(device)
    # Forward and generation deliberately use native MRoPE/DeepStack, not LLaVA positions.
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False, logits_to_keep=1)
        if not torch.isfinite(outputs.logits).all().item():
            raise RuntimeError("Native BF16 forward returned non-finite last-token logits.")
        del outputs
        report["finite_forward_passed"] = True
        generated = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            num_beams=1, use_cache=True,
        )
    torch.cuda.synchronize()
    suffix = generated[:, input_tokens:]
    answer = processor.batch_decode(suffix, skip_special_tokens=True)[0]
    report["generated_tokens"] = suffix.shape[-1]
    report["generated_answer"] = answer
    report["gpu"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["gpu"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    if suffix.shape[-1] == 0 or not answer.strip():
        raise RuntimeError("Qwen generation returned an empty answer.")
    report["generation_passed"] = True
    # This tests loading/inference, not reasoning accuracy or mask output.


def smoke(args):
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "stage": "native_qwen_environment_before_sam",
        "status": "running",
        "started_at": utc_now(),
        "model_dir": str(args.model_dir),
        "image": str(args.image),
        "prompt": args.prompt,
        "attention_implementation": "sdpa",
        "max_new_tokens": args.max_new_tokens,
    }
    exit_code = 0
    try:
        run_smoke(args, report)
        report["status"] = "passed"
    except Exception as exc:
        exit_code = 1
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    report.update(completed_at=utc_now(), elapsed_seconds=time.monotonic() - started)
    write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return exit_code


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and record an immutable HF revision.")
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--revision", default="main", help="HF revision; resolved to a commit SHA.")
    check = commands.add_parser("smoke", help="Offline BF16/SDPA native Qwen GPU smoke test.")
    check.add_argument("--model-dir", type=Path, required=True)
    check.add_argument("--image", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    check.add_argument("--prompt", default=DEFAULT_PROMPT)
    check.add_argument("--min-pixels", type=positive_int, default=16384)
    check.add_argument("--max-pixels", type=positive_int, default=1048576)
    check.add_argument("--max-seq-length", type=positive_int, default=4096)
    check.add_argument("--max-new-tokens", type=positive_int, default=128)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        if args.min_pixels > args.max_pixels:
            parser.error("--min-pixels cannot exceed --max-pixels")
        if args.max_new_tokens >= args.max_seq_length:
            parser.error("--max-new-tokens must be below --max-seq-length")
        args.model_dir = args.model_dir.expanduser().resolve()
        args.image = args.image.expanduser().resolve()
        args.report = args.report.expanduser().resolve()
        # A report must never overwrite the image or any model artifact.
        if args.report == args.image or args.report.is_relative_to(args.model_dir):
            parser.error("--report must be outside the model directory and not the input image")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "download":
        download_snapshot(args)
        return 0
    return smoke(args)


if __name__ == "__main__":
    sys.exit(main())
