import glob
import json
import os
import random
import re

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)


RAIL_REASONING_DECISION_PATTERN = re.compile(
    r"This is a (?P<switch>turnout|merge) switch\. "
    r"The right blade is (?P<right_state>open|closed) and the left blade is (?P<left_state>open|closed)\. "
    r"The open (?P<open_side>right|left) blade and the closed (?P<closed_side>right|left) blade together create a continuous rail "
    r"connection toward the (?P<ego_path>right-hand|left-hand) path and break continuity with the (?P<other_path>right-hand|left-hand) path\. "
    r"Therefore, the ego-path follows the (?P<final_path>right-hand|left-hand) path\."
)

RAIL_EGO_SIDE_TO_LABEL = {"left-hand": 0, "right-hand": 1}
RAIL_EGO_SIDE_IGNORE_INDEX = -100


def parse_rail_ego_side_label(text):
    match = RAIL_REASONING_DECISION_PATTERN.search(text)
    if match is None:
        return RAIL_EGO_SIDE_IGNORE_INDEX
    return RAIL_EGO_SIDE_TO_LABEL.get(
        match.group("final_path"),
        RAIL_EGO_SIDE_IGNORE_INDEX,
    )


class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        weight_map_dir_name="weight_maps",
        weight_map_weight=1.0,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        self.weight_map_dir_name = weight_map_dir_name
        self.weight_map_weight = weight_map_weight

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size, allow_upscale=False)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.img_to_ego_side_label = {}

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                )
            )
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))

        explanation_path = os.path.join(
            base_image_dir,
            "reason_seg",
            reason_seg_data,
            "explanatory",
            "train.json",
        )
        if os.path.exists(explanation_path):
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            with open(explanation_path) as f:
                items = json.load(f)
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }
                ego_side_label = parse_rail_ego_side_label(item["outputs"])
                if ego_side_label != RAIL_EGO_SIDE_IGNORE_INDEX:
                    self.img_to_ego_side_label[img_name] = ego_side_label

            print("len(self.img_to_explanation): ", len(self.img_to_explanation))
            print("len(self.img_to_ego_side_label): ", len(self.img_to_ego_side_label))
        else:
            self.img_to_explanation = {}

    def __len__(self):
        return self.samples_per_epoch

    def load_reason_seg_weight_map(self, image_path):
        if self.weight_map_weight <= 1.0:
            return None

        split_dir = os.path.dirname(image_path)
        dataset_dir = os.path.dirname(split_dir)
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        weight_map_path = os.path.join(
            dataset_dir, self.weight_map_dir_name, image_stem + "_weight.png"
        )
        if not os.path.exists(weight_map_path):
            return None

        weight_map = np.array(Image.open(weight_map_path))
        if weight_map.ndim == 3:
            weight_map = weight_map[:, :, 0]
        weight_mask = weight_map > 0

        pixel_weights = np.ones(weight_mask.shape, dtype=np.float32)
        pixel_weights[weight_mask] = self.weight_map_weight
        return torch.from_numpy(pixel_weights)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        reason_seg_weight_map = self.load_reason_seg_weight_map(image_path)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2
            else:
                choice = random.randint(0, 1)

        questions = []
        answers = []
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # add explanation if applicable
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1 and img_name in self.img_to_explanation:
                if choice == 0:  # [SEG] token
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:  # [SEG] token + text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    answer = "{} {}".format(answer, random.choice(self.answer_list))
                    questions[-1] = (
                        DEFAULT_IMAGE_TOKEN
                        + "\n"
                        + text
                        + " {}".format(random.choice(self.explanatory_question_list))
                    )
                    answers.append(answer)
                elif choice == 2:  # vanilla text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text
                    answers.append(answer)
                else:
                    raise ValueError("Not implemented yet.")
            else:
                answers.append(random.choice(self.answer_list))

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        image_name = image_path.split("/")[-1]
        if (
            self.explanatory != -1
            and image_name in self.img_to_explanation
            and choice == 2
        ):
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
            reason_seg_weight_maps = torch.empty(0)
        else:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
            if reason_seg_weight_map is not None:
                reason_seg_weight_maps = torch.stack(
                    [reason_seg_weight_map.clone() for _ in range(len(sampled_sents))],
                    dim=0,
                )
            else:
                reason_seg_weight_maps = torch.empty(0)

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
            reason_seg_weight_maps,
            self.img_to_ego_side_label.get(
                image_name,
                RAIL_EGO_SIDE_IGNORE_INDEX,
            ),
        )
