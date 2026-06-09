import glob
import os
import random
import json
from PIL import Image

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .rail_reasoning import (
    RAIL_REASONING_DECISION_GROUP_WEIGHTS,
    RAIL_REASONING_DECISION_PATTERN,
    SEG_TOKEN_CE_WEIGHT,
    build_rail_reasoning_decision_token_mask,
)
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from .vqa_dataset import VQADataset

def build_rail_reasoning_ce_weights(assistant_text, tokenizer, target_len):
    weights = torch.ones(target_len, dtype=torch.float32)

    match = RAIL_REASONING_DECISION_PATTERN.search(assistant_text)
    if match is None:
        return weights

    for group_name, group_weight in RAIL_REASONING_DECISION_GROUP_WEIGHTS.items():
        if group_weight <= 1.0:
            continue
        char_start, char_end = match.span(group_name)
        token_start = len(tokenizer(assistant_text[:char_start], add_special_tokens=False).input_ids)
        token_end = len(tokenizer(assistant_text[:char_end], add_special_tokens=False).input_ids)
        token_start = max(0, min(target_len, token_start))
        token_end = max(token_start, min(target_len, token_end))
        if token_end > token_start:
            weights[token_start:token_end] = group_weight

    return weights

def apply_token_sequence_weight(token_weights, token_ids, token_sequence, weight):
    if weight <= 1.0 or len(token_sequence) == 0:
        return

    sequence_len = len(token_sequence)
    last_start = len(token_ids) - sequence_len + 1
    for start_idx in range(max(0, last_start)):
        if token_ids[start_idx : start_idx + sequence_len] == token_sequence:
            token_weights[start_idx : start_idx + sequence_len] = weight


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, use_rail_reasoning_weighted_ce=False, use_seg_token_weighted_ce=False,
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    reason_seg_weight_maps_list = []
    conversation_is_rail_reasoning = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for sample in batch:
        if len(sample) == 10:
            (
                image_path,
                images,
                images_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
                inference,
            ) = sample
            reason_seg_weight_maps = torch.empty(0)
            is_rail_reasoning_sample = False
        elif len(sample) == 11:
            (
                image_path,
                images,
                images_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
                reason_seg_weight_maps,
                inference,
            ) = sample
            is_rail_reasoning_sample = f"{os.sep}ReasonSegRail{os.sep}" in image_path
            # print(
            #     "[rail path debug]",
            #     "image_path=", image_path,
            #     "is_rail_reasoning_sample=", is_rail_reasoning_sample,
            # )
        else:
            raise ValueError(f"Unexpected batch sample length: {len(sample)}")

        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        reason_seg_weight_maps_list.append(reason_seg_weight_maps.float())
        conversation_is_rail_reasoning.extend(
            [is_rail_reasoning_sample for _ in range(len(conversations))]
        )
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    ce_token_weights = torch.ones_like(targets, dtype=torch.float32)
    reasoning_decision_token_masks = torch.zeros_like(targets, dtype=torch.bool)
    seg_token_ids = tokenizer("[SEG]", add_special_tokens=False).input_ids

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
        
    for (
        conversation_idx,
        (conversation, input_id, target, token_weight, is_rail_reasoning),
    ) in enumerate(
        zip(
            conversation_list,
            input_ids,
            targets,
            ce_token_weights,
            conversation_is_rail_reasoning,
        )
    ):       
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            assistant_start = cur_len + instruction_len
            assistant_end = cur_len + round_len
            if is_rail_reasoning and use_rail_reasoning_weighted_ce:
                assistant_text = parts[1]
                assistant_token_count = assistant_end - assistant_start
                assistant_weights = build_rail_reasoning_ce_weights(
                    assistant_text,
                    tokenizer,
                    assistant_token_count,
                )
                token_weight[assistant_start:assistant_end] = assistant_weights

            if is_rail_reasoning:
                assistant_text = parts[1]
                assistant_token_count = assistant_end - assistant_start
                decision_token_mask = build_rail_reasoning_decision_token_mask(
                    assistant_text,
                    tokenizer,
                    assistant_token_count,
                )
                # if "[SEG]" in assistant_text:
                #     print(
                #         "[reason prompt debug]",
                #         "matched_decision_tokens=", int(decision_token_mask.sum()),
                #         "assistant_token_count=", assistant_token_count,
                #         "assistant_text=", repr(assistant_text),
                #     )
                reasoning_decision_token_masks[
                    conversation_idx, assistant_start:assistant_end
                ] |= decision_token_mask
            
            if use_seg_token_weighted_ce:
                assistant_token_ids = input_id[assistant_start:assistant_end].tolist()
                apply_token_sequence_weight(
                    token_weight[assistant_start:assistant_end],
                    assistant_token_ids,
                    seg_token_ids,
                    SEG_TOKEN_CE_WEIGHT,
                )

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
            ce_token_weights = ce_token_weights[:, :truncate_len]
            reasoning_decision_token_masks = reasoning_decision_token_masks[
                :, :truncate_len
            ]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "ce_token_weights": ce_token_weights,
        "reasoning_decision_token_masks": reasoning_decision_token_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "reason_seg_weight_maps_list": reason_seg_weight_maps_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class HybridDataset(torch.utils.data.Dataset):
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
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        reason_seg_rail_data="ReasonSegRail|train",
        explanatory=0.1,
        reason_seg_weight_map_dir_name="weight_maps",
        reason_seg_weight_map_weight=1.0,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        reason_seg_weight_map_dir_name,
                        reason_seg_weight_map_weight,
                    )
                )

            elif dataset == "reason_seg_rail":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_rail_data,
                        explanatory,
                        reason_seg_weight_map_dir_name,
                        reason_seg_weight_map_weight,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


def init_railsem_sem_seg_val(base_image_dir, split):
    railsem_data_root = os.path.join(base_image_dir, "RailSem19-SemSeg-LISA")
    with open(os.path.join(railsem_data_root, "config_v2.0.json")) as f:
        railsem_classes = json.load(f)["labels"]
    railsem_classes = np.array([x["readable"].lower() for x in railsem_classes])
    railsem_labels = sorted(
        glob.glob(
            os.path.join(railsem_data_root, split, "v2.0", "labels", "*.png")
        )
    )
    if len(railsem_labels) == 0:
        raise ValueError(
            f"No RailSem semantic-segmentation labels found under {railsem_data_root}/{split}/v2.0/labels"
        )
    railsem_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in railsem_labels
    ]
    return railsem_classes, railsem_images, railsem_labels


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3 and splits[0] == "sem_seg":
            _, ds, split = splits
            if ds != "railsem":
                raise ValueError(
                    f"Unsupported sem_seg validation dataset: {ds}. Use sem_seg|railsem|<split>."
                )
            self.sem_seg_classes, self.images, self.labels = init_railsem_sem_seg_val(
                self.base_image_dir, split
            )
            self.data_type = "sem_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size, allow_upscale=False)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

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
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        elif self.data_type == "sem_seg":
            image_path = self.images[idx]
            label_path = self.labels[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = np.array(Image.open(label_path))
            if self.ds == "railsem":
                if label.ndim == 3:
                    label = label[:, :, 0]

            sampled_class_ids = np.unique(label).tolist()
            if self.ignore_label in sampled_class_ids:
                sampled_class_ids.remove(self.ignore_label)
            if len(sampled_class_ids) == 0:
                return self.__getitem__((idx + 1) % len(self))
            sampled_sents = [self.sem_seg_classes[class_id] for class_id in sampled_class_ids]
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        elif self.data_type == "sem_seg":
            label = torch.from_numpy(label).long()
            masks = torch.stack([label == class_id for class_id in sampled_class_ids], dim=0)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )
