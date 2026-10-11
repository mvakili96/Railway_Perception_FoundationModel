"""Backend-neutral dataset contract; deliberately importable without ML packages."""

from dataclasses import dataclass, field
from typing import Any


def validate_sample_mode(mode):
    if mode not in ("llava", "structured"):
        raise ValueError(f"Unknown sample_mode={mode!r}; use 'llava' or 'structured'.")
    return mode


def render_llava_conversations(questions, answers):
    # Importing model.llava eagerly also imports its old language-model classes.
    # The Qwen path must never need those classes or the old Transformers API.
    from model.llava import conversation as conversation_lib

    conv = conversation_lib.default_conversation.copy()
    conversations = []
    for question, answer in zip(questions, answers):
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
    return conversations


@dataclass
class StructuredSegSample:
    image_path: str
    rgb_image: Any  # augmented uint8 HWC, before CLIP/SAM resizing
    sam_image: Any  # existing normalized/padded CHW SAM tensor
    questions: list
    answers: list
    masks: Any
    label: Any
    resize: tuple
    sampled_classes: list
    reason_seg_weight_maps: Any
    rail_ego_side_label: int = -100
    is_rail_reasoning: bool = False
    inference: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def original_size(self):
        return tuple(self.rgb_image.shape[:2])

    @property
    def mask_offsets(self):
        offsets = [0]
        for answer in self.answers:
            offsets.append(offsets[-1] + answer.count("[SEG]"))
        return offsets

    def validate(self):
        if not self.questions or len(self.questions) != len(self.answers):
            raise ValueError(f"{self.image_path}: question/answer pairs must be nonempty and aligned.")
        if any(not isinstance(text, str) for text in self.questions + self.answers):
            raise ValueError(f"{self.image_path}: questions and answers must be strings.")
        if len(self.rgb_image.shape) != 3 or self.rgb_image.shape[-1] != 3:
            raise ValueError(f"{self.image_path}: raw RGB must have shape [H, W, 3].")
        if str(self.rgb_image.dtype) != "uint8":
            raise ValueError(f"{self.image_path}: raw RGB must remain uint8, not SAM-normalized.")
        if len(self.masks.shape) != 3 or tuple(self.masks.shape[1:]) != self.original_size:
            raise ValueError(f"{self.image_path}: masks must remain at the original image size.")
        if self.mask_offsets[-1] != self.masks.shape[0]:
            raise ValueError(
                f"{self.image_path}: {self.mask_offsets[-1]} answer [SEG] tokens but "
                f"{self.masks.shape[0]} masks; mask ordering cannot be inferred safely."
            )
        if tuple(self.label.shape) != self.original_size:
            raise ValueError(f"{self.image_path}: label size differs from raw RGB.")
        if len(self.sam_image.shape) != 3 or self.sam_image.shape[0] != 3:
            raise ValueError(f"{self.image_path}: SAM input must be CHW.")
        if self.reason_seg_weight_maps.numel() and (
            tuple(self.reason_seg_weight_maps.shape) != tuple(self.masks.shape)
        ):
            raise ValueError(f"{self.image_path}: weight maps must match the masks.")
        return self
