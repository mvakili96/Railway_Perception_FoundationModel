"""Qwen-specific trainability, losses, device transfer and ZeRO configuration."""

from dataclasses import asdict, dataclass
import torch
from torch.nn import functional as F

from model.segmentation_losses import dice_loss, sigmoid_ce_loss


@dataclass(frozen=True)
class TrainingPolicy:
    lr: float = 1e-4
    vision_lr: float = 1e-5
    vision_last_blocks: int = 8
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    ce_loss_weight: float = 1.0
    bce_loss_weight: float = 2.0
    dice_loss_weight: float = 1.0
    rail_ego_side_loss_weight: float = 3.0
    boundary_bce_band_width: int = 0
    boundary_bce_weight: float = 1.0

    def __post_init__(self):
        if self.lr <= 0 or self.vision_lr <= 0 or self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("Learning rates and LoRA sizes must be positive.")
        if self.vision_last_blocks < 0 or not 0 <= self.lora_dropout < 1:
            raise ValueError("Invalid vision/LoRA policy.")
        if min(self.ce_loss_weight, self.bce_loss_weight, self.dice_loss_weight,
               self.rail_ego_side_loss_weight, self.boundary_bce_band_width) < 0:
            raise ValueError("Loss weights and boundary width cannot be negative.")


def configure_training(model, policy):
    """Apply PEFT first, then restore the explicitly chosen non-LoRA parameters."""
    from peft import LoraConfig, get_peft_model
    if hasattr(model.qwen, "peft_config"):
        raise ValueError("Refusing to wrap Qwen in PEFT twice.")
    native = model.native_qwen
    targets = [name for name, module in native.named_modules()
               if name.startswith("model.language_model.layers.") and
               name.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")]
    if len(targets) != native.config.text_config.num_hidden_layers * 4:
        raise ValueError("Expected exactly four language-attention projections per text layer.")
    model.qwen = get_peft_model(native, LoraConfig(r=policy.lora_r,
        lora_alpha=policy.lora_alpha, lora_dropout=policy.lora_dropout,
        target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    native = model.native_qwen
    native.get_input_embeddings().seg_row.requires_grad_(True)
    native.get_output_embeddings().seg_row.requires_grad_(True)
    visual = native.model.visual
    count = min(policy.vision_last_blocks, len(visual.blocks))
    if count:
        for block in visual.blocks[-count:]:
            block.requires_grad_(True)
    visual.merger.requires_grad_(True)
    visual.deepstack_merger_list.requires_grad_(True)
    model.configure_sam_trainability()
    for module in (model.text_projection, model.rail_ego_side_head, model.seg_query_bridge):
        if module is not None:
            module.requires_grad_(True)
    groups = {key: {"params": [], "names": [], "lr": lr, "group_name": key}
              for key, lr in (("language_lora", policy.lr), ("vision", policy.vision_lr),
                              ("bridge_and_seg", policy.lr), ("sam", policy.lr))}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("qwen.") and ".lora_" in name:
            if ".language_model.layers." not in name:
                raise ValueError("LoRA must not target Qwen vision modules.")
            key = "language_lora"
        elif name.startswith("qwen.") and ".visual." in name:
            key = "vision"
        elif name.startswith("sam."):
            key = "sam"
        elif name.startswith(("text_projection.", "rail_ego_side_head.", "seg_query_bridge.")) or name.endswith(".seg_row"):
            key = "bridge_and_seg"
        else:
            raise ValueError(f"Unexpected trainable base parameter: {name}")
        groups[key]["params"].append(parameter)
        groups[key]["names"].append(name)
    manifest = [{"name": key, "lr": value["lr"], "parameter_count": sum(p.numel() for p in value["params"]),
                 "parameter_names": value["names"]} for key, value in groups.items()]
    optimizer_groups = [{key: value for key, value in group.items() if key != "names"}
                        for group in groups.values() if group["params"]]
    parameters = [p for group in optimizer_groups for p in group["params"]]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Duplicate optimizer parameters.")
    return optimizer_groups, manifest


def deepspeed_config(batch_size, accumulation, world_size, policy, total_steps,
                     offload=False, warmup_steps=100):
    if min(batch_size, accumulation, world_size, total_steps) <= 0:
        raise ValueError("Batch sizes and schedule must be positive.")
    # Explicit per-group warmup preserves the 10x vision LR separation.
    rates = [policy.lr, policy.vision_lr, policy.lr, policy.lr]
    config = {"train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": accumulation,
        "train_batch_size": batch_size * accumulation * world_size,
        "bf16": {"enabled": True}, "gradient_clipping": 1.0,
        "optimizer": {"type": "AdamW", "params": {"lr": policy.lr,
            "weight_decay": 0.0, "betas": [0.9, 0.95], "torch_adam": not offload}},
        "scheduler": {"type": "WarmupDecayLR", "params": {"total_num_steps": total_steps,
            "warmup_min_lr": [0.0] * 4, "warmup_max_lr": rates,
            "warmup_num_steps": max(2, min(warmup_steps, total_steps - 1)), "warmup_type": "linear"}},
        "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
            "reduce_scatter": True, "reduce_bucket_size": 200000, "allgather_bucket_size": 200000}}
    if offload:
        config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    return config


def move_batch(batch, device, dtype=torch.bfloat16):
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value
    result = {key: move(value) for key, value in batch.items()}
    for key in ("images", "pixel_values"):
        result[key] = result[key].to(dtype=dtype)
    return result


def compute_losses(output, batch, policy, ce_chunk_size=128):
    """Identical supervision/weighting; FP32 loss arithmetic, bounded CE temporaries."""
    if ce_chunk_size <= 0:
        raise ValueError("CE chunk size must be positive.")
    zero = output["keepalive_loss"].float()
    labels = batch["labels"][:, 1:]
    weights = batch["ce_token_weights"][:, 1:].float()
    valid = labels.ne(-100).nonzero(as_tuple=False)
    ce_sum, weight_sum = zero, zero.detach().new_zeros(())
    fields = {field: [zero, zero.detach().new_zeros(()), zero.detach().new_zeros(())]
              for field in ("switch", "right_state")}
    for start in range(0, len(valid), ce_chunk_size):
        rows, columns = valid[start:start + ce_chunk_size].unbind(1)
        logits = output["logits"][rows, columns].float()
        targets, token_weights = labels[rows, columns], weights[rows, columns]
        token_ce = F.cross_entropy(logits, targets, reduction="none")
        ce_sum = ce_sum + (token_ce * token_weights).sum()
        weight_sum = weight_sum + token_weights.sum()
        correct = logits.argmax(-1).eq(targets)
        for field in fields:
            selected = batch.get(f"rail_{field}_token_masks")
            if selected is not None:
                mask = selected[rows, columns + 1].bool()
                fields[field][0] = fields[field][0] + token_ce[mask].sum()
                fields[field][1] = fields[field][1] + correct[mask].sum()
                fields[field][2] = fields[field][2] + mask.sum()
    ce = ce_sum / weight_sum.clamp_min(1)
    bce, dice, mask_count = zero, zero, 0
    targets_list = batch["masks_list"]
    if len(targets_list) != len(output["pred_masks"]):
        raise ValueError("Mask lists do not match source images.")
    for index, (prediction, target) in enumerate(zip(output["pred_masks"], targets_list)):
        if prediction.shape != target.shape:
            raise ValueError("SEG count/order differs from ground-truth masks.")
        count = target.shape[0]
        if not count:
            continue
        maps = batch.get("reason_seg_weight_maps_list")
        pixel_weights = maps[index] if maps is not None and maps[index].numel() else None
        if pixel_weights is not None and pixel_weights.shape != target.shape:
            raise ValueError("Weight-map dimensions differ from masks.")
        bce = bce + sigmoid_ce_loss(prediction.float(), target.float(), count,
            policy.boundary_bce_band_width, policy.boundary_bce_weight, pixel_weights) * count
        dice = dice + dice_loss(prediction.float(), target.float(), count) * count
        mask_count += count
    bce = bce / (mask_count + 1e-8) * policy.bce_loss_weight
    dice = dice / (mask_count + 1e-8) * policy.dice_loss_weight
    ego = zero
    side_labels = torch.repeat_interleave(batch["rail_ego_side_labels"], output["seg_token_counts"])
    if side_labels.shape[0] != output["rail_ego_side_logits"].shape[0]:
        raise ValueError("Ego labels must align with SEG events.")
    selected = side_labels.ne(-100)
    if bool(selected.any()):
        ego = ego + F.cross_entropy(output["rail_ego_side_logits"][selected].float(), side_labels[selected])
    ce = ce * policy.ce_loss_weight
    losses = {"ce_loss": ce, "mask_bce_loss": bce, "mask_dice_loss": dice,
              "mask_loss": bce + dice, "rail_ego_side_loss": ego,
              "loss": ce + bce + dice + ego * policy.rail_ego_side_loss_weight + zero}
    for field, (total, correct, count) in fields.items():
        losses.update({f"{field}_ce_sum": total, f"{field}_correct": correct,
                       f"{field}_token_count": count})
    return losses
