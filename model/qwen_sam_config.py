"""Portable composite settings; importable without torch/Transformers."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


CONFIG_NAME = "qwen_sam_config.json"


@dataclass(frozen=True)
class QwenSAMConfig:
    schema_version: int = 1
    seg_prompt_bridge_type: str = "single"
    seg_query_num_heads: int = 8
    seg_query_num_hidden_layers: int = 4
    sam_model_type: str = "vit_h"
    train_mask_decoder: bool = True
    train_sam_prompt_encoder: bool = True
    train_sam_last_blocks: int = 16
    train_sam_neck: bool = False
    train_sam_patch_embed: bool = False
    seg_bridge_numerics_debug: bool = False
    seg_token_id: int | None = None
    qwen_vocab_capacity: int | None = None
    source_qwen_revision: str | None = None
    sam_init_source: str | None = None
    sam_init_path: str | None = None

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported Qwen/SAM composite schema.")
        if self.seg_prompt_bridge_type not in ("single", "four_query"):
            raise ValueError("seg_prompt_bridge_type must be single or four_query.")
        if self.sam_model_type != "vit_h":
            raise ValueError("The agreed SAM architecture is ViT-H.")
        if self.seg_query_num_heads <= 0 or 256 % self.seg_query_num_heads:
            raise ValueError("SAM's 256 prompt dimensions must divide evenly across heads.")
        if self.seg_query_num_hidden_layers <= 0 or self.train_sam_last_blocks < 0:
            raise ValueError("Invalid layer counts.")
        if self.sam_init_source not in (None, "pretrained", "rail_checkpoint"):
            raise ValueError("sam_init_source must be pretrained or rail_checkpoint.")
        if self.seg_token_id is not None and self.seg_token_id < 0:
            raise ValueError("Invalid SEG token identity.")
        if self.qwen_vocab_capacity is not None:
            if self.qwen_vocab_capacity <= 0 or (
                self.seg_token_id is not None and self.seg_token_id >= self.qwen_vocab_capacity
            ):
                raise ValueError("SEG must fit within the saved vocabulary capacity.")

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CONFIG_NAME).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, directory):
        return cls(**json.loads((Path(directory) / CONFIG_NAME).read_text()))


def select_sam_state_dict(source, expected_keys, source_type):
    """Select SAM tensors only; reject ambiguous/partial/partitioned checkpoints."""
    if source_type not in ("pretrained", "rail_checkpoint"):
        raise ValueError("An explicit sam_init_source is required for a fresh run.")
    if not isinstance(source, dict):
        raise ValueError("SAM checkpoint must contain a complete tensor state dictionary.")
    for container in ("state_dict", "module"):
        if container in source and isinstance(source[container], dict):
            source = source[container]
            break
    selected = {}
    expected = set(expected_keys)
    for key, value in source.items():
        normalized = key
        for prefix in ("module.", "base_model.model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        if source_type == "rail_checkpoint":
            for prefix in ("model.visual_model.", "visual_model.", "sam."):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    break
            else:
                continue
        if normalized not in expected:
            if source_type == "pretrained":
                raise ValueError(f"Unexpected original SAM tensor: {key}")
            continue
        if normalized in selected:
            raise ValueError(f"Duplicate SAM tensor after prefix normalization: {normalized}")
        selected[normalized] = value
    missing = expected - selected.keys()
    if missing:
        raise ValueError("Incomplete SAM checkpoint; no weights loaded. Missing: " +
                         ", ".join(sorted(missing)[:20]))
    return selected
