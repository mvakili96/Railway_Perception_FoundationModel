"""Native Qwen3-VL collator for existing structured segmentation samples."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from transformers import AddedToken

from .segmentation_sample import StructuredSegSample
from .qwen_token_alignment import (
    ASSISTANT_END, assistant_positions, expand_image_for_alignment,
    reasoning_positions, render_qwen_pair,
)
from .rail_reasoning import (
    RAIL_REASONING_DECISION_PATTERN, RAIL_REASONING_DECISION_GROUP_WEIGHTS,
    RAIL_REASONING_MASK_ONLY_ANSWERS, SEG_TOKEN_CE_WEIGHT,
)


IGNORE_INDEX = -100


@dataclass(frozen=True)
class QwenProcessingConfig:
    min_pixels: int = 16384
    max_pixels: int = 1048576
    max_seq_length: int = 4096

    def __post_init__(self):
        if not 0 < self.min_pixels <= self.max_pixels or self.max_seq_length <= 0:
            raise ValueError("Pixel budgets and sequence limit must be positive and ordered.")

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "qwen_processing_config.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, directory):
        return cls(**json.loads((Path(directory) / "qwen_processing_config.json").read_text()))


def register_seg_token(tokenizer):
    """Tokenizer-only registration; Stage 3 handles embeddings without shrinking vocab."""
    tokenizer.add_tokens([AddedToken("[SEG]", normalized=False, special=False)], special_tokens=False)
    ids = tokenizer("[SEG]", add_special_tokens=False).input_ids
    if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=True) != "[SEG]":
        raise ValueError("[SEG] must be one ordinary added token, visible during decoding.")
    return ids[0]


class QwenSegCollator:
    def __init__(self, processor, config=None, use_rail_reasoning_weighted_ce=False,
                 use_seg_token_weighted_ce=False):
        self.processor = processor
        self.config = config or QwenProcessingConfig()
        self.tokenizer = processor.tokenizer
        if not self.tokenizer.is_fast:
            raise ValueError("Qwen collator requires the native fast tokenizer for exact offsets.")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Native Qwen tokenizer must supply a padding token.")
        self.seg_token_id = register_seg_token(self.tokenizer)
        self.use_rail_reasoning_weighted_ce = use_rail_reasoning_weighted_ce
        self.use_seg_token_weighted_ce = use_seg_token_weighted_ce

    def _conversation(self, sample, question, answer):
        text, assistant_start = render_qwen_pair(
            self.processor, question, answer, inference=sample.inference,
        )
        native = self.processor(
            text=[text], images=[sample.rgb_image], return_tensors="pt",
            padding=False, truncation=False, return_offsets_mapping=True,
            min_pixels=self.config.min_pixels, max_pixels=self.config.max_pixels,
        )
        ids = native["input_ids"][0]
        if ids.numel() > self.config.max_seq_length:
            raise ValueError(
                f"{sample.image_path}: {ids.numel()} Qwen tokens exceed "
                f"max_seq_length={self.config.max_seq_length}; "
                f"grid={native['image_grid_thw'].tolist()}; no truncation was performed."
            )
        grid = native["image_grid_thw"]
        if tuple(grid.shape) != (1, 3):
            raise ValueError(f"{sample.image_path}: expected one image grid per conversation.")
        visual_tokens = int(grid.prod(-1).item()) // self.processor.image_processor.merge_size**2
        if ids.eq(self.processor.image_token_id).sum().item() != visual_tokens:
            raise ValueError(f"{sample.image_path}: native image grid/token counts disagree.")
        expanded = expand_image_for_alignment(text, self.processor.image_token, visual_tokens)
        # Tokenize the *exact expanded native string*, never guessed image offsets.
        alignment = self.tokenizer(
            expanded, add_special_tokens=False, return_offsets_mapping=True,
        )
        offsets = [tuple(pair) for pair in native.pop("offset_mapping")[0].tolist()]
        id_list = ids.tolist()
        if alignment.input_ids != id_list or list(map(tuple, alignment.offset_mapping)) != offsets:
            raise ValueError(f"{sample.image_path}: native processor and alignment tokenization differ.")
        labels = torch.full_like(ids, IGNORE_INDEX)
        weights = torch.ones(ids.shape, dtype=torch.float32)
        groups = {}
        positions = []
        seg_positions = []
        span = None
        if not sample.inference:
            # All visual placeholders precede the assistant answer.
            char_start = assistant_start + (visual_tokens - 1) * len(self.processor.image_token)
            char_end = char_start + len(answer) + len(ASSISTANT_END)
            positions = assistant_positions(id_list, offsets, expanded, char_start, char_end, self.tokenizer)
            labels[positions] = ids[positions]
            if id_list[positions[-1]] != self.tokenizer.convert_tokens_to_ids(ASSISTANT_END):
                raise ValueError("The assistant end token must be supervised.")
            seg_positions = [i for i in positions if id_list[i] == self.seg_token_id]
            if len(seg_positions) != answer.count("[SEG]"):
                raise ValueError("Supervised [SEG] tokens do not match the answer text.")
            if sample.is_rail_reasoning:
                groups = reasoning_positions(
                    answer, offsets, char_start, id_list, self.tokenizer,
                    RAIL_REASONING_DECISION_PATTERN, RAIL_REASONING_MASK_ONLY_ANSWERS,
                )
                supervised = set(positions)
                for name, indices in groups.items():
                    if not set(indices) <= supervised:
                        raise ValueError(f"Rail field {name!r} overlaps ignored labels.")
                    if self.use_rail_reasoning_weighted_ce:
                        weights[indices] = RAIL_REASONING_DECISION_GROUP_WEIGHTS.get(name, 1.0)
            if self.use_seg_token_weighted_ce:
                weights[seg_positions] = SEG_TOKEN_CE_WEIGHT
            span = {"char_start": char_start, "char_end": char_end,
                    "token_start": positions[0], "token_end": positions[-1] + 1}
        return {
            "input_ids": ids, "labels": labels, "ce_token_weights": weights,
            "pixel_values": native["pixel_values"], "image_grid_thw": grid,
            "seg_positions": seg_positions, "groups": groups, "assistant_span": span,
            "conversation": text, "visual_tokens": visual_tokens,
        }

    def __call__(self, batch):
        if not batch or any(not isinstance(sample, StructuredSegSample) for sample in batch):
            raise ValueError("QwenSegCollator requires a nonempty batch of StructuredSegSample objects.")
        for sample in batch:
            sample.validate()
        if len({sample.inference for sample in batch}) != 1:
            raise ValueError("Cannot mix prompt-only inference and teacher-forced training samples.")
        rows = []
        offsets = [0]
        conversation_to_image = []
        answers = []
        ego_labels = []
        for image_index, sample in enumerate(batch):
            for question, answer in zip(sample.questions, sample.answers):
                try:
                    rows.append(self._conversation(sample, question, answer))
                except ValueError as exc:
                    raise ValueError(f"{sample.image_path}: {exc}") from exc
                answers.append(answer)
                conversation_to_image.append(image_index)
                ego_labels.append(sample.rail_ego_side_label if sample.is_rail_reasoning else IGNORE_INDEX)
            offsets.append(len(rows))
        pad = torch.nn.utils.rnn.pad_sequence
        # Decoder-only generation must read the last real prompt token, not PAD.
        # Training stays right-padded so verified answer/metric indices are unchanged.
        padding_side = "left" if batch[0].inference else "right"
        ids = pad([row["input_ids"] for row in rows], batch_first=True,
                  padding_value=self.tokenizer.pad_token_id, padding_side=padding_side)
        labels = pad([row["labels"] for row in rows], batch_first=True,
                     padding_value=IGNORE_INDEX, padding_side=padding_side)
        attention_mask = pad([torch.ones_like(row["input_ids"]) for row in rows],
                             batch_first=True, padding_value=0, padding_side=padding_side)
        metric_masks = {key: torch.zeros_like(ids, dtype=torch.bool) for key in (
            "reasoning_decision_token_masks", "rail_switch_token_masks", "rail_right_state_token_masks",
        )}
        for index, row in enumerate(rows):
            for key, group in (("reasoning_decision_token_masks", "open_side"),
                               ("rail_switch_token_masks", "switch"),
                               ("rail_right_state_token_masks", "right_state")):
                metric_masks[key][index, row["groups"].get(group, [])] = True
        return {
            "input_ids": ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": torch.cat([row["pixel_values"] for row in rows], dim=0),
            "image_grid_thw": torch.cat([row["image_grid_thw"] for row in rows], dim=0),
            "ce_token_weights": pad([row["ce_token_weights"] for row in rows],
                                    batch_first=True, padding_value=1.0, padding_side=padding_side),
            "images": torch.stack([sample.sam_image for sample in batch]),
            "image_paths": [sample.image_path for sample in batch],
            "masks_list": [sample.masks.float() for sample in batch],
            "label_list": [sample.label for sample in batch],
            "resize_list": [sample.resize for sample in batch],
            "original_size_list": [sample.original_size for sample in batch],
            "reason_seg_weight_maps_list": [sample.reason_seg_weight_maps.float() for sample in batch],
            "rail_ego_side_labels": torch.tensor(ego_labels, dtype=torch.long),
            "offset": torch.tensor(offsets, dtype=torch.long),
            "conversation_to_image": torch.tensor(conversation_to_image, dtype=torch.long),
            "mask_offsets_list": [sample.mask_offsets for sample in batch],
            "questions_list": [sample.questions for sample in batch],
            "sampled_classes_list": [sample.sampled_classes for sample in batch],
            "metadata_list": [sample.metadata for sample in batch],
            "answer_list": answers,
            "conversation_list": [row["conversation"] for row in rows],
            "assistant_spans": [row["assistant_span"] for row in rows],
            "reasoning_group_positions": [row["groups"] for row in rows],
            "seg_token_positions": [row["seg_positions"] for row in rows],
            "seg_token_id": self.seg_token_id,
            "visual_token_counts": [row["visual_tokens"] for row in rows],
            "inference": batch[0].inference,
            **metric_masks,
        }
