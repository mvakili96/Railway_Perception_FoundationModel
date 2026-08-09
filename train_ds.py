import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from functools import partial
import inspect
import torch


def init_wandb(args):
    if not args.use_wandb or not args.is_main_process or args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging was requested with --use_wandb, but wandb is not installed. "
            "Install it in the environment or add a pip --target path to PYTHONPATH."
        ) from exc

    wandb_dir = args.wandb_dir or args.log_dir
    os.makedirs(wandb_dir, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or args.exp_name,
        dir=wandb_dir,
        mode=args.wandb_mode,
        config=vars(args),
    )


def wandb_log(wandb_run, metrics, step):
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)

_sig = inspect.signature(torch.nn.Module.register_forward_pre_hook)
if ('prepend' not in _sig.parameters) or ('with_kwargs' not in _sig.parameters):
    _orig = torch.nn.Module.register_forward_pre_hook
    def _compat(self, hook, *args, **kwargs):
        # Older torch doesn't accept these kwargs; ignore them.
        return _orig(self, hook)
    torch.nn.Module.register_forward_pre_hook = _compat

import deepspeed
import cv2
import numpy as np
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.rail_reasoning import (
    RAIL_REASONING_DECISION_GROUP_WEIGHTS,
    RAIL_REASONING_DECISION_PATTERN,
)
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, AverageMeter,
                         ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)


EPOCH_REASONING_PROMPT = (
    "Based on the blade positions in this switch, which route corresponds to "
    "the route the train takes? Please respond with segmentation mask and "
    "explain why."
)


def log_switch_mask_debug(
    input_dict,
    tokenizer,
    max_samples,
    expected_right_state_weight,
    right_states_logged,
):
    """Print and verify the switch and right-state weighted target tokens."""
    if max_samples <= 0:
        return 0

    labels = input_dict["labels"]
    input_ids = input_dict["input_ids"]
    switch_masks = input_dict["rail_switch_token_masks"].bool()
    right_state_masks = input_dict["rail_right_state_token_masks"].bool()
    token_weights = input_dict["ce_token_weights"]
    conversations = input_dict["conversation_list"]
    logged = 0

    for row_idx in range(labels.shape[0]):
        selected_mask = switch_masks[row_idx] & labels[row_idx].ne(IGNORE_INDEX)
        positions = selected_mask.nonzero(as_tuple=False).flatten()
        right_state_positions = right_state_masks[row_idx].nonzero(
            as_tuple=False
        ).flatten()
        if positions.numel() == 0 or right_state_positions.numel() == 0:
            continue

        if labels[row_idx, right_state_positions].eq(IGNORE_INDEX).any():
            raise ValueError(
                "Right-state debug mask overlaps ignored labels: "
                f"row={row_idx} positions={right_state_positions.tolist()}"
            )

        right_state_expected_match = re.search(
            r"This is a (?:turnout|merge) switch\. "
            r"The right blade is (open|closed)\b",
            conversations[row_idx],
        )
        right_state_expected = (
            right_state_expected_match.group(1)
            if right_state_expected_match is not None
            else None
        )
        if right_state_expected is None:
            raise ValueError(
                "Right-state debug mask has no canonical open/closed target: "
                f"row={row_idx} conversation={conversations[row_idx]!r}"
            )

        missing_states = {"open", "closed"} - right_states_logged
        if missing_states and right_state_expected not in missing_states:
            continue

        token_ids = labels[row_idx, positions].tolist()
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        expected_match = re.search(
            r"This is a (turnout|merge) switch",
            conversations[row_idx],
        )
        expected = expected_match.group(1) if expected_match is not None else None
        weights = token_weights[row_idx, positions].tolist()

        context_start = max(0, int(positions[0]) - 5)
        context_end = min(input_ids.shape[1], int(positions[-1]) + 6)
        context_ids = input_ids[row_idx, context_start:context_end].tolist()
        context = tokenizer.decode(context_ids, skip_special_tokens=False)

        print(
            "[switch token debug] "
            f"expected={expected!r} decoded={decoded!r} "
            f"match={expected is not None and decoded.strip() == expected} "
            f"token_ids={token_ids} positions={positions.tolist()} "
            f"weights={weights} context={context!r}",
            flush=True,
        )

        right_state_token_ids = labels[
            row_idx, right_state_positions
        ].tolist()
        right_state_input_ids = input_ids[
            row_idx, right_state_positions
        ].tolist()
        right_state_decoded = tokenizer.decode(
            right_state_token_ids,
            skip_special_tokens=False,
        )
        right_state_weights = token_weights[
            row_idx, right_state_positions
        ].tolist()
        right_state_matches = (
            right_state_expected is not None
            and right_state_decoded.strip() == right_state_expected
        )
        input_ids_match = right_state_input_ids == right_state_token_ids
        weights_match = all(
            abs(weight - expected_right_state_weight) <= 1e-6
            for weight in right_state_weights
        )

        context_start = max(0, int(right_state_positions[0]) - 5)
        context_end = min(
            input_ids.shape[1],
            int(right_state_positions[-1]) + 6,
        )
        context_ids = input_ids[row_idx, context_start:context_end].tolist()
        context = tokenizer.decode(context_ids, skip_special_tokens=False)

        print(
            "[right_state token debug] "
            f"expected={right_state_expected!r} "
            f"decoded={right_state_decoded!r} "
            f"match={right_state_matches} "
            f"input_ids_match={input_ids_match} "
            f"token_ids={right_state_token_ids} "
            f"positions={right_state_positions.tolist()} "
            f"weights={right_state_weights} "
            f"expected_weight={expected_right_state_weight} "
            f"weights_match={weights_match} context={context!r}",
            flush=True,
        )

        if not right_state_matches or not input_ids_match or not weights_match:
            raise ValueError(
                "Right-state weighted-token verification failed: "
                f"row={row_idx} expected={right_state_expected!r} "
                f"decoded={right_state_decoded!r} "
                f"input_ids_match={input_ids_match} "
                f"weights={right_state_weights} "
                f"expected_weight={expected_right_state_weight}"
            )

        right_states_logged.add(right_state_expected)
        logged += 1
        if logged >= max_samples:
            break

    return logged


def build_epoch_reasoning_manifest(args):
    """Resolve the fixed Rail reasoning probe set once before training."""
    json_path = os.path.abspath(os.path.expanduser(args.epoch_reasoning_json))
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            "Epoch reasoning inference JSON was not found: {}".format(json_path)
        )

    dataset_parts = args.reason_seg_rail_data.split("|")
    if len(dataset_parts) != 2:
        raise ValueError(
            "--reason_seg_rail_data must have the form DATASET|SPLIT when "
            "--epoch_reasoning_inference is enabled; got {!r}".format(
                args.reason_seg_rail_data
            )
        )
    dataset_name, split = dataset_parts
    image_dir = os.path.join(
        args.dataset_dir,
        "reason_seg",
        dataset_name,
        split,
    )

    with open(json_path, "r") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(
            "Epoch reasoning inference JSON must contain a list: {}".format(
                json_path
            )
        )

    selected = []
    seen_images = set()
    missing_images = []
    image_pattern = re.compile(r"^rs(?P<index>\d+)\.[^.]+$")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                "Every epoch reasoning JSON entry must be an object; got {!r}".format(
                    record
                )
            )
        image_name = os.path.basename(str(record.get("image", "")))
        image_match = image_pattern.fullmatch(image_name)
        if image_match is None:
            continue
        image_index = int(image_match.group("index"))
        if image_index >= args.epoch_reasoning_image_index_limit:
            continue
        if image_name in seen_images:
            raise ValueError(
                "Duplicate image in epoch reasoning inference subset: {}".format(
                    image_name
                )
            )
        seen_images.add(image_name)

        ground_truth = record.get("outputs")
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            raise ValueError(
                "Epoch reasoning inference entry has no non-empty outputs: {}".format(
                    image_name
                )
            )
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            missing_images.append(image_path)

        selected.append(
            {
                "image": image_name,
                "image_index": image_index,
                "image_path": image_path,
                "ground_truth": ground_truth.strip(),
            }
        )

    if missing_images:
        preview = ", ".join(missing_images[:10])
        suffix = "" if len(missing_images) <= 10 else " ..."
        raise FileNotFoundError(
            "Missing {} epoch reasoning inference images: {}{}".format(
                len(missing_images), preview, suffix
            )
        )
    if not selected:
        raise ValueError(
            "No rs<number> images with index below {} were found in {}".format(
                args.epoch_reasoning_image_index_limit,
                json_path,
            )
        )

    selected.sort(key=lambda item: (item["image_index"], item["image"]))
    return selected, json_path, image_dir


def build_epoch_reasoning_prompt(args):
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + EPOCH_REASONING_PROMPT
    if args.use_mm_start_end:
        replacement = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        user_prompt = user_prompt.replace(DEFAULT_IMAGE_TOKEN, replacement)

    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], "")
    return conv.get_prompt()


def _normalize_reasoning_text(text):
    return " ".join(text.strip().split())


def _parse_rail_reasoning(text):
    match = RAIL_REASONING_DECISION_PATTERN.search(
        _normalize_reasoning_text(text)
    )
    return match.groupdict() if match is not None else None


def _epoch_reasoning_summary(results):
    successful = [item for item in results if "error" not in item]
    normalized_predictions = [
        _normalize_reasoning_text(item["prediction"]) for item in successful
    ]
    exact_matches = sum(
        prediction == _normalize_reasoning_text(item["ground_truth"])
        for item, prediction in zip(successful, normalized_predictions)
    )

    decision_fields = ("switch", "right_state", "left_state", "final_path")
    decision_correct = {field: 0 for field in decision_fields}
    ground_truth_switch_counts = {}
    predicted_switch_counts = {}
    canonical_predictions = 0
    for item in successful:
        predicted = _parse_rail_reasoning(item["prediction"])
        expected = _parse_rail_reasoning(item["ground_truth"])
        if expected is not None:
            expected_switch = expected["switch"]
            ground_truth_switch_counts[expected_switch] = (
                ground_truth_switch_counts.get(expected_switch, 0) + 1
            )
        if predicted is None:
            continue
        canonical_predictions += 1
        predicted_switch = predicted["switch"]
        predicted_switch_counts[predicted_switch] = (
            predicted_switch_counts.get(predicted_switch, 0) + 1
        )
        if expected is None:
            continue
        for field in decision_fields:
            decision_correct[field] += predicted[field] == expected[field]

    total = len(results)
    summary = {
        "sample_count": total,
        "successful_count": len(successful),
        "error_count": total - len(successful),
        "unique_prediction_count": len(set(normalized_predictions)),
        "collapsed_to_one_prediction": (
            len(successful) > 1 and len(set(normalized_predictions)) == 1
        ),
        "exact_match_count": exact_matches,
        "canonical_prediction_count": canonical_predictions,
        "ground_truth_switch_counts": ground_truth_switch_counts,
        "predicted_switch_counts": predicted_switch_counts,
    }
    for field in decision_fields:
        summary[field + "_correct"] = decision_correct[field]
        summary[field + "_accuracy"] = (
            decision_correct[field] / total if total else 0.0
        )
    return summary


def run_epoch_reasoning_inference(
    model_engine,
    tokenizer,
    clip_image_processor,
    manifest,
    manifest_path,
    image_dir,
    epoch,
    checkpoint_saved,
    args,
):
    """Greedily decode explanations from the exact live DeepSpeed model."""
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else args.global_rank
    world_size = torch.distributed.get_world_size() if distributed else 1
    if distributed:
        torch.distributed.barrier()

    zero_stage_attr = getattr(model_engine, "zero_optimization_stage", None)
    zero_stage = zero_stage_attr() if callable(zero_stage_attr) else zero_stage_attr
    if zero_stage is not None and int(zero_stage) >= 3:
        raise RuntimeError(
            "Epoch reasoning inference currently supports DeepSpeed ZeRO stages "
            "0-2 only; rank-local generation is unsafe with ZeRO-3 partitioned "
            "parameters."
        )

    live_model = model_engine.module
    was_training = live_model.training
    prompt = build_epoch_reasoning_prompt(args)
    prompt_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        return_tensors="pt",
    ).unsqueeze(0)
    device = torch.device("cuda", args.local_rank)
    prompt_ids = prompt_ids.to(device=device)
    # The LLaVA cached-decoding path dereferences attention_mask.shape on every
    # token after the first.  Some Transformers/PEFT combinations do not infer
    # this mask through the wrapper, so pass the unpadded prompt mask explicitly.
    prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    if args.precision == "bf16":
        image_dtype = torch.bfloat16
    elif args.precision == "fp16":
        image_dtype = torch.float16
    else:
        image_dtype = torch.float32

    local_manifest = manifest[rank::world_size]
    local_results = []
    model_engine.eval()
    try:
        with torch.inference_mode():
            for sample in local_manifest:
                result = {
                    "image": sample["image"],
                    "image_index": sample["image_index"],
                    "ground_truth": sample["ground_truth"],
                    "rank": rank,
                }
                error_stage = "read_image"
                try:
                    image_np = cv2.imread(sample["image_path"])
                    if image_np is None:
                        raise ValueError(
                            "cv2.imread returned None for {}".format(
                                sample["image_path"]
                            )
                        )
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    error_stage = "clip_preprocess"
                    image_clip = clip_image_processor.preprocess(
                        image_np,
                        return_tensors="pt",
                    )["pixel_values"].to(device=device, dtype=image_dtype)

                    error_stage = "generate"
                    generated = live_model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_attention_mask,
                        images=image_clip,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=args.epoch_reasoning_max_new_tokens,
                        use_cache=True,
                        synced_gpus=False,
                    )
                    error_stage = "decode"
                    sequences = (
                        generated.sequences
                        if hasattr(generated, "sequences")
                        else generated
                    )
                    generated_ids = sequences[0, prompt_ids.shape[1] :]
                    if generated_ids.lt(0).any():
                        raise ValueError(
                            "Generated suffix contains a negative token ID"
                        )
                    result["prediction"] = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    ).strip()
                except Exception as exc:
                    result["error"] = "{}: {}".format(
                        type(exc).__name__,
                        exc,
                    )
                    result["error_stage"] = error_stage
                    result["traceback"] = traceback.format_exc(limit=20)
                local_results.append(result)
                if "error" in result:
                    # One failure is enough to fail the probe. Avoid repeating a
                    # systemic generation error for every image on this rank.
                    break
    finally:
        model_engine.train(was_training)

    if distributed:
        gathered_results = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_results, local_results)
        results = [item for rank_items in gathered_results for item in rank_items]
    else:
        results = local_results
    results.sort(key=lambda item: (item["image_index"], item["image"]))

    local_error = any("error" in item for item in local_results)
    error_flag = torch.tensor(
        [int(local_error)],
        dtype=torch.int32,
        device=device,
    )
    if distributed:
        torch.distributed.all_reduce(
            error_flag,
            op=torch.distributed.ReduceOp.MAX,
        )

    if args.is_main_process:
        global_step = getattr(model_engine, "global_steps", None)
        active_adapter = getattr(live_model, "active_adapter", None)
        header = {
            "epoch_index": epoch,
            "epoch_number": epoch + 1,
            "global_step": global_step,
            "checkpoint_saved_before_probe": bool(checkpoint_saved),
            "prompt": EPOCH_REASONING_PROMPT,
            "manifest": manifest_path,
            "image_dir": image_dir,
            "image_index_limit": args.epoch_reasoning_image_index_limit,
            "sample_count": len(manifest),
            "max_new_tokens": args.epoch_reasoning_max_new_tokens,
            "version": args.version,
            "model_class": type(live_model).__name__,
            "base_model_class": type(
                live_model.get_base_model()
                if hasattr(live_model, "get_base_model")
                else live_model
            ).__name__,
            "active_adapter": (
                str(active_adapter) if active_adapter is not None else None
            ),
            "deepspeed_zero_stage": zero_stage,
        }
        print(
            "[epoch reasoning header] " + json.dumps(header, ensure_ascii=False),
            flush=True,
        )
        for result in results:
            payload = {
                "epoch_index": epoch,
                "epoch_number": epoch + 1,
                "global_step": global_step,
                **result,
            }
            print(
                "[epoch reasoning result] "
                + json.dumps(payload, ensure_ascii=False),
                flush=True,
            )
        print(
            "[epoch reasoning summary] "
            + json.dumps(
                {
                    "epoch_index": epoch,
                    "epoch_number": epoch + 1,
                    "global_step": global_step,
                    **_epoch_reasoning_summary(results),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if distributed:
        torch.distributed.barrier()
    if error_flag.item():
        raise RuntimeError(
            "Epoch reasoning inference failed for one or more samples; inspect "
            "the [epoch reasoning result] error records above."
        )
    return results


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )

    parser.add_argument("--hf_merged_model", action="store_true", default=False,
                    help="Model is already merged (HF release like xinlai/LISA-7B-v1); skip SAM/LLaVA init.")


    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--reason_seg_rail_data", default="ReasonSegRail|train", type=str)
    parser.add_argument(
        "--val_dataset",
        default="ReasonSeg|val",
        type=str,
        help="Validation dataset, e.g. ReasonSeg|val, refcoco|unc|val, or sem_seg|railsem|validation",
    )
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Log training and validation metrics to Weights & Biases on global rank 0.",
    )
    parser.add_argument("--wandb_project", default="lisa-rail", type=str)
    parser.add_argument("--wandb_entity", default="", type=str)
    parser.add_argument("--wandb_name", default="", type=str)
    parser.add_argument("--wandb_dir", default="", type=str)
    parser.add_argument(
        "--wandb_mode",
        default="online",
        type=str,
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument(
        "--use_rail_reasoning_weighted_ce",
        action="store_true",
        default=False,
        help="Enable hardcoded per-slot CE weights for Rail ReasonSeg explanations.",
    )

    parser.add_argument(
        "--use_seg_token_weighted_ce",
        action="store_true",
        default=False,
        help="Enable hardcoded extra CE weight for the [SEG] token.",
    )

    parser.add_argument(
        "--rail_ego_side_loss_weight",
        default=0.0,
        type=float,
        help="Auxiliary CE loss weight for predicting left/right ego-side from the [SEG] hidden state.",
    )

    parser.add_argument(
        "--use_rail_reasoning_prompt_tokens",
        action="store_true",
        default=False,
        help="Append Rail reasoning open-side token hidden states to the [SEG] SAM prompt.",
    )

    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)

    parser.add_argument(
        "--boundary_bce_band_width",
        default=0,
        type=int,
        help="Half-width in pixels of the GT-boundary band for BCE reweighting. 0 disables it.",
    )
    parser.add_argument(
        "--boundary_bce_weight",
        default=1.0,
        type=float,
        help="Extra BCE weight applied inside the GT-boundary band. 1.0 disables it.",
    )

    parser.add_argument(
        "--reason_seg_weight_map_dir_name",
        default="weight_maps",
        type=str,
        help="Sibling folder name under each ReasonSeg dataset root that stores .png pixel-weight maps.",
    )
    parser.add_argument(
        "--reason_seg_weight_map_weight",
        default=1.0,
        type=float,
        help="Extra pixel weight applied where the ReasonSeg weight-map value is nonzero. 1.0 disables it.",
    )

    parser.add_argument(
        "--rail_counterfactual_flip_prob",
        default=0.0,
        type=float,
        help="Probability of horizontally flipping a Rail ReasonSeg sample and swapping its left/right semantics.",
    )

    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument(
        "--switch_ce_debug_samples",
        default=8,
        type=int,
        help=(
            "Number of paired switch/right-state weighted-target examples to "
            "decode and verify in the training log; 0 disables it."
        ),
    )
    parser.add_argument(
        "--epoch_reasoning_inference",
        action="store_true",
        default=False,
        help=(
            "After every epoch, greedily decode Rail explanations from the exact "
            "live in-memory model and print them to stdout."
        ),
    )
    parser.add_argument(
        "--epoch_reasoning_json",
        default="train.json",
        type=str,
        help=(
            "Explanatory JSON used by --epoch_reasoning_inference. Relative paths "
            "are resolved from the training process working directory."
        ),
    )
    parser.add_argument(
        "--epoch_reasoning_image_index_limit",
        default=1000,
        type=int,
        help=(
            "Select rs<number> images whose numeric filename index is strictly "
            "below this value for epoch reasoning inference."
        ),
    )
    parser.add_argument(
        "--epoch_reasoning_max_new_tokens",
        default=128,
        type=int,
        help="Maximum tokens generated per epoch reasoning inference sample.",
    )
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)

    parser.add_argument(
        "--train_sam_neck",
        action="store_true",
        default=False,
        help="Unfreeze the SAM image encoder neck.",
    )
    
    parser.add_argument(
        "--train_sam_patch_embed",
        action="store_true",
        default=False,
        help="Unfreeze the SAM image encoder patch embedding.",
    )

    parser.add_argument(
        "--train_sam_prompt_encoder",
        action="store_true",
        default=False,
        help="Unfreeze the SAM prompt encoder.",
    )

    parser.add_argument(
        "--train_sam_last_blocks",
        default=0,
        type=int,
        help="Number of final SAM image encoder transformer blocks to unfreeze.",
    )

    parser.add_argument(
        "--train_mm_projector",
        action="store_true",
        default=False,
        help="Unfreeze LLaVA's CLIP-to-LLM mm_projector.",
    )

    parser.add_argument(
        "--train_clip_last_blocks",
        default=0,
        type=int,
        help="Number of final CLIP vision blocks feeding the selected LLaVA feature to unfreeze.",
    )
    parser.add_argument(
        "--clip_lr",
        default=1e-5,
        type=float,
        help="Learning rate for trainable CLIP vision parameters.",
    )

    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    args.global_rank = int(os.environ.get("RANK", "0"))
    args.world_size = int(
        os.environ.get("WORLD_SIZE", str(max(torch.cuda.device_count(), 1)))
    )
    args.is_main_process = args.global_rank == 0

    epoch_reasoning_manifest = None
    epoch_reasoning_manifest_path = None
    epoch_reasoning_image_dir = None
    epoch_reasoning_clip_processor = None
    if args.epoch_reasoning_inference:
        if args.epoch_reasoning_image_index_limit <= 0:
            raise ValueError(
                "--epoch_reasoning_image_index_limit must be positive"
            )
        if args.epoch_reasoning_max_new_tokens <= 0:
            raise ValueError(
                "--epoch_reasoning_max_new_tokens must be positive"
            )
        (
            epoch_reasoning_manifest,
            epoch_reasoning_manifest_path,
            epoch_reasoning_image_dir,
        ) = build_epoch_reasoning_manifest(args)
        epoch_reasoning_clip_processor = CLIPImageProcessor.from_pretrained(
            args.vision_tower
        )
        if args.is_main_process:
            print(
                "[epoch reasoning setup] "
                + json.dumps(
                    {
                        "manifest": epoch_reasoning_manifest_path,
                        "image_dir": epoch_reasoning_image_dir,
                        "image_index_limit": (
                            args.epoch_reasoning_image_index_limit
                        ),
                        "sample_count": len(epoch_reasoning_manifest),
                        "prompt": EPOCH_REASONING_PROMPT,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if args.is_main_process:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    wandb_run = init_wandb(args)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    vision_pretrained = None if args.hf_merged_model else args.vision_pretrained
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "train_sam_neck": args.train_sam_neck,
        "train_sam_patch_embed": args.train_sam_patch_embed,
        "train_sam_prompt_encoder": args.train_sam_prompt_encoder,
        "train_sam_last_blocks": args.train_sam_last_blocks,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "rail_ego_side_loss_weight": args.rail_ego_side_loss_weight,    
        "boundary_bce_band_width": args.boundary_bce_band_width,
        "boundary_bce_weight": args.boundary_bce_weight,
        "use_rail_reasoning_prompt_tokens": args.use_rail_reasoning_prompt_tokens,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    if args.hf_merged_model:
        model.ce_loss_weight   = args.ce_loss_weight
        model.dice_loss_weight = args.dice_loss_weight
        model.bce_loss_weight  = args.bce_loss_weight
        model.rail_ego_side_loss_weight = args.rail_ego_side_loss_weight
        model.config.rail_ego_side_loss_weight = args.rail_ego_side_loss_weight
        model.boundary_bce_band_width = args.boundary_bce_band_width
        model.boundary_bce_weight = args.boundary_bce_weight
        model.use_rail_reasoning_prompt_tokens = args.use_rail_reasoning_prompt_tokens
        model.config.use_rail_reasoning_prompt_tokens = args.use_rail_reasoning_prompt_tokens
        model.out_dim          = args.out_dim
        model.seg_token_idx    = args.seg_token_idx

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()


    # --- INIT BLOCK (fixed) ---

    # Always build/wire the CLIP vision wrapper (safe with merged HF models)
    model.get_model().initialize_vision_modules(model.get_model().config)

    # Move the vision tower to the right device/dtype
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    # IMPORTANT:
    # Only re-init LISA modules when you're training from *backbones*.
    # Skip this when starting from a merged HF model to avoid overwriting.
    if (not args.hf_merged_model) and (not args.eval_only):
        model.get_model().initialize_lisa_modules(model.get_model().config)
    # --- END INIT BLOCK ---




    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))


    # Re-enable intentionally trainable modules after PEFT wraps the model.
    sam_block_indices = []
    sam_block_prefix = "visual_model.image_encoder.blocks."
    for n, _ in model.named_parameters():
        if sam_block_prefix in n:
            block_idx = n.split(sam_block_prefix, 1)[1].split(".", 1)[0]
            if block_idx.isdigit():
                sam_block_indices.append(int(block_idx))
    num_sam_blocks = max(sam_block_indices) + 1 if len(sam_block_indices) > 0 else 0
    num_train_sam_blocks = max(
        0, min(num_sam_blocks, args.train_sam_last_blocks)
    )
    sam_train_block_start = num_sam_blocks - num_train_sam_blocks

    clip_block_indices = []
    clip_block_prefix = "vision_tower.vision_tower.vision_model.encoder.layers."
    for n, _ in model.named_parameters():
        if clip_block_prefix in n:
            block_idx = n.split(clip_block_prefix, 1)[1].split(".", 1)[0]
            if block_idx.isdigit():
                clip_block_indices.append(int(block_idx))
    num_clip_blocks = max(clip_block_indices) + 1 if len(clip_block_indices) > 0 else 0
    num_train_clip_blocks = max(
        0, min(num_clip_blocks, args.train_clip_last_blocks)
    )
    clip_train_block_start = num_clip_blocks
    clip_train_block_end = -1
    if num_train_clip_blocks > 0:
        selected_layer = getattr(vision_tower, "select_layer", -1)
        selected_hidden_idx = (
            num_clip_blocks + 1 + selected_layer
            if selected_layer < 0
            else selected_layer
        )
        clip_train_block_end = max(0, min(num_clip_blocks - 1, selected_hidden_idx - 1))
        clip_train_block_start = max(
            0, clip_train_block_end - num_train_clip_blocks + 1
        )
        if args.is_main_process:
            print(
                "training CLIP vision blocks {}-{} with clip_lr={}".format(
                    clip_train_block_start, clip_train_block_end, args.clip_lr
                )
            )


    trainable_name_keys = [
        "lm_head",
        "embed_tokens",
        "mask_decoder",
        "text_hidden_fcs",
        "heatmap_head",
    ]
    for n, p in model.named_parameters():
        train_this_param = any(x in n for x in trainable_name_keys)
        if args.train_sam_neck and "visual_model.image_encoder.neck" in n:
            train_this_param = True

        if args.train_sam_patch_embed and "visual_model.image_encoder.patch_embed" in n:
            train_this_param = True

        if args.train_sam_prompt_encoder and "visual_model.prompt_encoder" in n:
            train_this_param = True

        if args.train_mm_projector and "mm_projector" in n:
            train_this_param = True

        if args.rail_ego_side_loss_weight > 0 and "rail_ego_side_head" in n:
            train_this_param = True

        if num_train_clip_blocks > 0 and clip_block_prefix in n:
            block_idx = n.split(clip_block_prefix, 1)[1].split(".", 1)[0]
            if (
                block_idx.isdigit()
                and clip_train_block_start <= int(block_idx) <= clip_train_block_end
            ):
                train_this_param = True

        if num_train_sam_blocks > 0 and sam_block_prefix in n:
            block_idx = n.split(sam_block_prefix, 1)[1].split(".", 1)[0]
            if block_idx.isdigit() and int(block_idx) >= sam_train_block_start:
                train_this_param = True

        if train_this_param:
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    if args.is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            "trainable params after manual unfreeze: {} || all params: {} || trainable%: {:.4f}".format(
                trainable_params, total_params, 100 * trainable_params / total_params
            )
        )

        if wandb_run is not None:
            wandb_run.summary["trainable_params"] = trainable_params
            wandb_run.summary["total_params"] = total_params
            wandb_run.summary["trainable_percent"] = (
                100 * trainable_params / total_params
            )

    clip_trainable_params = []
    base_trainable_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if num_train_clip_blocks > 0 and clip_block_prefix in n:
            clip_trainable_params.append(p)
        else:
            base_trainable_params.append(p)

    optimizer_param_groups = [{"params": base_trainable_params, "lr": args.lr}]
    warmup_min_lr = 0
    warmup_max_lr = args.lr
    if len(clip_trainable_params) > 0:
        optimizer_param_groups.append(
            {"params": clip_trainable_params, "lr": args.clip_lr}
        )
        warmup_min_lr = [0, 0]
        warmup_max_lr = [args.lr, args.clip_lr]
    if args.is_main_process:
        base_trainable_count = sum(p.numel() for p in base_trainable_params)
        clip_trainable_count = sum(p.numel() for p in clip_trainable_params)
        print(
            "optimizer param groups: base_params={} clip_params={}".format(
                base_trainable_count,
                clip_trainable_count,
            )
        )

        if wandb_run is not None:
            wandb_run.summary["base_trainable_params"] = base_trainable_count
            wandb_run.summary["clip_trainable_params"] = clip_trainable_count
    world_size = args.world_size
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        reason_seg_rail_data=args.reason_seg_rail_data,
        explanatory=args.explanatory,
        reason_seg_weight_map_dir_name=args.reason_seg_weight_map_dir_name,
        reason_seg_weight_map_weight=args.reason_seg_weight_map_weight,
        rail_counterfactual_flip_prob=args.rail_counterfactual_flip_prob,
    )

    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
                "torch_adam": True,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": warmup_min_lr,
                "warmup_max_lr": warmup_max_lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e5,
            "allgather_bucket_size": 2e5,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=optimizer_param_groups,
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
            use_rail_reasoning_weighted_ce=args.use_rail_reasoning_weighted_ce,
            use_seg_token_weighted_ce=args.use_seg_token_weighted_ce,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
                use_rail_reasoning_weighted_ce=args.use_rail_reasoning_weighted_ce,
                use_seg_token_weighted_ce=args.use_seg_token_weighted_ce,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        giou, ciou = validate(val_loader, model_engine, 0, writer, args, wandb_run)
        if wandb_run is not None:
            wandb_run.finish()
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
            wandb_run,
            tokenizer,
        )

        is_best = False
        if args.no_eval == False:
            giou, ciou = validate(
                val_loader,
                model_engine,
                epoch,
                writer,
                args,
                wandb_run,
            )
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        will_save = args.no_eval or is_best
        checkpoint_saved = False
        if will_save:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.is_main_process:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir, ignore_errors=True)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            checkpoint_saved = True

        if args.epoch_reasoning_inference:
            run_epoch_reasoning_inference(
                model_engine=model_engine,
                tokenizer=tokenizer,
                clip_image_processor=epoch_reasoning_clip_processor,
                manifest=epoch_reasoning_manifest,
                manifest_path=epoch_reasoning_manifest_path,
                image_dir=epoch_reasoning_image_dir,
                epoch=epoch,
                checkpoint_saved=checkpoint_saved,
                args=args,
            )

    if wandb_run is not None:
        wandb_run.finish()

def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
    wandb_run,
    tokenizer,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    switch_ce_losses = AverageMeter("SwitchCE", ":.4f")
    switch_accuracies = AverageMeter("SwitchAcc", ":.4f")
    right_state_ce_losses = AverageMeter("RightStateCE", ":.4f")
    right_state_accuracies = AverageMeter("RightStateAcc", ":.4f")
    rail_ego_side_losses = AverageMeter("RailEgoLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            switch_ce_losses,
            switch_accuracies,
            right_state_ce_losses,
            right_state_accuracies,
            rail_ego_side_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        log_step = epoch * args.steps_per_epoch + global_step
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            if args.is_main_process:
                debug_samples_logged = getattr(
                    args,
                    "switch_ce_debug_samples_logged",
                    0,
                )
                debug_samples_remaining = max(
                    0,
                    args.switch_ce_debug_samples - debug_samples_logged,
                )
                if debug_samples_remaining > 0:
                    right_states_logged = getattr(
                        args,
                        "right_state_debug_states_logged",
                        None,
                    )
                    if right_states_logged is None:
                        right_states_logged = set()
                        args.right_state_debug_states_logged = (
                            right_states_logged
                        )
                    expected_right_state_weight = (
                        RAIL_REASONING_DECISION_GROUP_WEIGHTS["right_state"]
                        if args.use_rail_reasoning_weighted_ce
                        else 1.0
                    )
                    newly_logged = log_switch_mask_debug(
                        input_dict,
                        tokenizer,
                        debug_samples_remaining,
                        expected_right_state_weight,
                        right_states_logged,
                    )
                    args.switch_ce_debug_samples_logged = (
                        debug_samples_logged + newly_logged
                    )
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            switch_ce = output_dict["switch_ce"]
            switch_token_count = int(output_dict["switch_token_count"].item())
            switch_accuracy = output_dict["switch_accuracy"]
            right_state_ce = output_dict["right_state_ce"]
            right_state_token_count = int(
                output_dict["right_state_token_count"].item()
            )
            right_state_accuracy = output_dict["right_state_accuracy"]
            rail_ego_side_loss = output_dict["rail_ego_side_loss"]

            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            if switch_token_count > 0:
                switch_ce_losses.update(switch_ce.item(), switch_token_count)
                switch_accuracies.update(
                    switch_accuracy.item(),
                    switch_token_count,
                )
            if right_state_token_count > 0:
                right_state_ce_losses.update(
                    right_state_ce.item(),
                    right_state_token_count,
                )
                right_state_accuracies.update(
                    right_state_accuracy.item(),
                    right_state_token_count,
                )
            rail_ego_side_losses.update(
                rail_ego_side_loss.item(),
                input_dict["images"].size(0),
            )
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                switch_ce_losses.all_reduce()
                switch_accuracies.all_reduce()
                right_state_ce_losses.all_reduce()
                right_state_accuracies.all_reduce()
                rail_ego_side_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.is_main_process:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, log_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, log_step)
                writer.add_scalar("train/switch_ce", switch_ce_losses.avg, log_step)
                writer.add_scalar(
                    "train/switch_accuracy",
                    switch_accuracies.avg,
                    log_step,
                )
                writer.add_scalar(
                    "train/switch_token_count", switch_ce_losses.count, log_step
                )
                print(
                    "[switch metrics] "
                    f"step={log_step} ce={switch_ce_losses.avg:.6f} "
                    f"accuracy={switch_accuracies.avg:.6f} "
                    f"token_count={switch_ce_losses.count:g}",
                    flush=True,
                )
                writer.add_scalar(
                    "train/right_state_ce",
                    right_state_ce_losses.avg,
                    log_step,
                )
                writer.add_scalar(
                    "train/right_state_accuracy",
                    right_state_accuracies.avg,
                    log_step,
                )
                writer.add_scalar(
                    "train/right_state_token_count",
                    right_state_ce_losses.count,
                    log_step,
                )
                print(
                    "[right-state metrics] "
                    f"step={log_step} ce={right_state_ce_losses.avg:.6f} "
                    f"accuracy={right_state_accuracies.avg:.6f} "
                    f"token_count={right_state_ce_losses.count:g}",
                    flush=True,
                )
                writer.add_scalar(
                    "train/rail_ego_side_loss",
                    rail_ego_side_losses.avg,
                    log_step,
                )
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, log_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, log_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, log_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, log_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, log_step
                )

                train_metrics = {
                    "train/loss": losses.avg,
                    "train/ce_loss": ce_losses.avg,
                    "train/switch_ce": switch_ce_losses.avg,
                    "train/switch_accuracy": switch_accuracies.avg,
                    "train/switch_token_count": switch_ce_losses.count,
                    "train/right_state_ce": right_state_ce_losses.avg,
                    "train/right_state_accuracy": right_state_accuracies.avg,
                    "train/right_state_token_count": right_state_ce_losses.count,
                    "train/rail_ego_side_loss": rail_ego_side_losses.avg,
                    "train/mask_bce_loss": mask_bce_losses.avg,
                    "train/mask_dice_loss": mask_dice_losses.avg,
                    "train/mask_loss": mask_losses.avg,
                    "metrics/total_secs_per_batch": batch_time.avg,
                    "metrics/data_secs_per_batch": data_time.avg,
                    "epoch": epoch,
                }
                if torch.cuda.is_available():
                    train_metrics["metrics/cuda_mem_allocated_gb"] = (
                        torch.cuda.memory_allocated() / 1024**3
                    )
                    train_metrics["metrics/cuda_max_mem_allocated_gb"] = (
                        torch.cuda.max_memory_allocated() / 1024**3
                    )
                wandb_log(wandb_run, train_metrics, log_step)

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            switch_ce_losses.reset()
            switch_accuracies.reset()
            right_state_ce_losses.reset()
            right_state_accuracies.reset()
            rail_ego_side_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.is_main_process:
                writer.add_scalar("train/lr", curr_lr[0], log_step)
                writer.add_scalar("train/lr_base", curr_lr[0], log_step)
                lr_metrics = {
                    "train/lr": curr_lr[0],
                    "train/lr_base": curr_lr[0],
                    "epoch": epoch,
                }
                if len(curr_lr) > 1:
                    writer.add_scalar("train/lr_clip", curr_lr[1], log_step)
                    lr_metrics["train/lr_clip"] = curr_lr[1]
                wandb_log(wandb_run, lr_metrics, log_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args, wandb_run):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict.pop("rail_right_state_token_masks", None)
        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.is_main_process:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        wandb_log(
            wandb_run,
            {
                "val/giou": giou,
                "val/ciou": ciou,
                "epoch": epoch,
            },
            (epoch + 1) * args.steps_per_epoch,
        )
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])

