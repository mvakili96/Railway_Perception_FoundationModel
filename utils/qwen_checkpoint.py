"""Explicit, complete Qwen/SAM training-checkpoint metadata and reconstruction."""

from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import torch
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from model.QwenSAM import QwenSAM
from model.qwen_sam_config import QwenSAMConfig
from model.segment_anything import build_sam_vit_h
from utils.qwen_processing import QwenProcessingConfig
from utils.qwen_training import TrainingPolicy, configure_training
from utils.qwen_checkpoint_selection import DEFAULT_BEST_CHECKPOINT_METRIC


METADATA_NAME = "training_config.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_metadata(run_dir):
    value = json.loads((Path(run_dir) / METADATA_NAME).read_text())
    if value.get("schema_version") != 1 or value.get("checkpoint_type") != "deepspeed_zero2_full":
        raise ValueError("Not a complete Qwen/SAM training run.")
    return value


def save_run_metadata(run_dir, model, policy, settings, parameter_manifest):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.processor.save_pretrained(run_dir / "processor")
    model.processing_config.save_pretrained(run_dir / "processor")
    model.native_qwen.config.save_pretrained(run_dir / "native_config")
    model.native_qwen.generation_config.save_pretrained(run_dir / "native_config")
    model.config.save_pretrained(run_dir)
    write_json(run_dir / "parameter_groups.json", parameter_manifest)
    write_json(run_dir / METADATA_NAME, {"schema_version": 1,
        "checkpoint_type": "deepspeed_zero2_full", "policy": asdict(policy), "settings": settings,
        "composite": asdict(model.config), "processing": asdict(model.processing_config)})


def check_resume_settings(metadata, settings):
    # Epoch count is part of the LR schedule: extending it requires an explicit
    # new experiment, not silently resetting or changing the restored scheduler.
    saved = dict(metadata["settings"])
    if "merger_lr" not in metadata.get("policy", {}) and "merger_lr" not in saved and "merger_lr" in settings:
        # Before split groups existed, every merger used the saved vision LR.
        # Normalize only this newly exposed setting, without changing the run.
        saved["merger_lr"] = saved.get("vision_lr", TrainingPolicy().vision_lr)
    if "lora_lr" not in metadata.get("policy", {}) and "lora_lr" not in saved and "lora_lr" in settings:
        # Earlier runs shared their main LR between language LoRA and SAM/bridge.
        # Only allow the historical rate when resuming, not a new experiment.
        saved["lora_lr"] = saved.get("lr", TrainingPolicy().lr)
    if "best_checkpoint_metric" not in saved and "best_checkpoint_metric" in settings:
        # Runs created before this option always tracked the best validation GIoU.
        saved["best_checkpoint_metric"] = DEFAULT_BEST_CHECKPOINT_METRIC
    if saved != settings:
        differing = [key for key in set(saved) | set(settings)
                     if saved.get(key) != settings.get(key)]
        raise ValueError("Resume configuration differs: " + ", ".join(sorted(differing)))


def reconstruct_training_model(run_dir, dtype=torch.bfloat16):
    """Create the exact architecture without requiring original Qwen/SAM files.

    Callers MUST immediately restore a strict complete model state. This is not
    pretrained initialization. CPU construction avoids meta-only nonpersistent
    rotary buffers and is intentional: request sufficient host RAM for export.
    """
    run_dir = Path(run_dir)
    metadata = read_metadata(run_dir)
    config = Qwen3VLConfig.from_pretrained(run_dir / "native_config", local_files_only=True)
    config._attn_implementation = "sdpa"
    processor = AutoProcessor.from_pretrained(run_dir / "processor", local_files_only=True)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        qwen = Qwen3VLForConditionalGeneration(config)
        sam = build_sam_vit_h(checkpoint=None)
        model = QwenSAM(qwen, sam, processor, QwenSAMConfig(**metadata["composite"]),
                        QwenProcessingConfig(**metadata["processing"]), _defer_validation=True)
    finally:
        torch.set_default_dtype(previous_dtype)
    from transformers import GenerationConfig
    model.native_qwen.generation_config = GenerationConfig.from_pretrained(
        run_dir / "native_config", local_files_only=True)
    policy_settings = dict(metadata["policy"])
    policy_settings.setdefault("lora_lr", policy_settings["lr"])
    legacy_vision_group = "merger_lr" not in policy_settings
    if legacy_vision_group:
        policy_settings["merger_lr"] = policy_settings["vision_lr"]
    groups, manifest = configure_training(model, TrainingPolicy(**policy_settings),
                                         legacy_vision_group=legacy_vision_group)
    saved_manifest = json.loads((run_dir / "parameter_groups.json").read_text())
    if saved_manifest != manifest:
        raise ValueError("Reconstructed trainability differs from saved checkpoint.")
    return model, groups, metadata


def resolve_checkpoint_tag(run_dir, tag=None):
    checkpoint_dir = Path(run_dir) / "checkpoints"
    if tag is None:
        tag = (checkpoint_dir / "latest").read_text().strip()
    if not tag or Path(tag).name != tag or tag in (".", ".."):
        raise ValueError("Checkpoint tag must be one directory name.")
    if not (checkpoint_dir / tag).is_dir():
        raise FileNotFoundError(checkpoint_dir / tag)
    return tag


def restore_for_export(run_dir, tag=None, dtype=torch.bfloat16):
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    tag = resolve_checkpoint_tag(run_dir, tag)
    model, _, metadata = reconstruct_training_model(run_dir, dtype)
    state = get_fp32_state_dict_from_zero_checkpoint(str(Path(run_dir) / "checkpoints"),
        tag=tag, exclude_frozen_parameters=False)
    # Includes frozen vocabulary, Qwen prefixes and SAM, not just LoRA/adapters.
    assert_complete_state(model, state)
    model.load_state_dict(state, strict=True)
    del state
    model.validate_auxiliary_parameters()
    return model, metadata, tag


def assert_complete_state(model, state):
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Full checkpoint key coverage differs; missing=" +
            str(sorted(set(expected) - set(state))) + "; unexpected=" + str(sorted(set(state) - set(expected))))
    for name, value in state.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"Checkpoint tensor shape differs: {name}")


def check_resume_model_file(model, run_dir, tag):
    # Avoid a bridge's legacy missing-gate fallback: reject incomplete state
    # BEFORE DeepSpeed delegates to module.load_state_dict. mmap avoids another
    # full host-RAM copy while checking tensor keys/shapes.
    files = list((Path(run_dir) / "checkpoints" / tag).glob("*model_states.pt"))
    if len(files) != 1:
        raise ValueError("Expected one non-model-parallel ZeRO-2 model-state file.")
    checkpoint = torch.load(files[0], map_location="cpu", mmap=True, weights_only=False)
    assert_complete_state(model, checkpoint["module"])


def prune_completed_checkpoints(run_dir, keep, protected_tags=()):
    """Only this workflow's committed epoch tags, and never latest/best tags."""
    if keep == 0:
        return []
    if keep < 0:
        raise ValueError("Checkpoint retention cannot be negative.")
    read_metadata(run_dir)  # ownership/schema guard
    root = (Path(run_dir) / "checkpoints").resolve()
    latest = resolve_checkpoint_tag(run_dir)
    protected = set(protected_tags) | {latest}
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and
                        not path.is_symlink() and re.fullmatch(r"epoch_[0-9]{3,}", path.name))
    removed = []
    for path in candidates[:-keep]:
        if path.name in protected:
            continue
        if path.resolve().parent != root:
            raise ValueError("Checkpoint deletion target escaped managed directory.")
        shutil.rmtree(path)
        removed.append(path.name)
    return removed
