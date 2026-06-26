from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)
from model.llava.constants import IGNORE_INDEX

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    boundary_band_width: int = 0,
    boundary_weight: float = 1.0,
    pixel_weights: torch.Tensor = None,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    if pixel_weights is not None and pixel_weights.numel() > 0:
        loss = loss * pixel_weights.to(loss.dtype)

    if boundary_band_width > 0 and boundary_weight > 1.0:
        targets_4d = targets.unsqueeze(1).float()
        kernel_size = 2 * boundary_band_width + 1
        dilated_targets = F.max_pool2d(
            targets_4d,
            kernel_size=kernel_size,
            stride=1,
            padding=boundary_band_width,
        )
        eroded_targets = -F.max_pool2d(
            -targets_4d,
            kernel_size=kernel_size,
            stride=1,
            padding=boundary_band_width,
        )
        boundary_band = (dilated_targets - eroded_targets) > 0

        pixel_weights = torch.ones_like(loss)
        pixel_weights = pixel_weights.masked_fill(
            boundary_band.squeeze(1), boundary_weight
        )
        loss = loss * pixel_weights

    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def bootstrapped_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    # Bootstrapped BCE: keep only hard pixels per mask.
    bootstrap_ratio = 0.25
    bootstrap_thresh = 0.1

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1)

    k = max(1, int(loss.shape[1] * bootstrap_ratio))
    sorted_loss, _ = torch.sort(loss, dim=1, descending=True)
    topk_loss = sorted_loss[:, :k]

    thresholded_loss = torch.where(
        topk_loss > bootstrap_thresh,
        topk_loss,
        torch.zeros_like(topk_loss),
    )
    valid_counts = (topk_loss > bootstrap_thresh).sum(dim=1, keepdim=True)
    fallback_loss = topk_loss.mean(dim=1)
    bootstrapped_loss = torch.where(
        valid_counts.squeeze(1) > 0,
        thresholded_loss.sum(dim=1) / valid_counts.squeeze(1).clamp_min(1),
        fallback_loss,
    )

    loss = bootstrapped_loss.sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.train_sam_neck = kwargs.get("train_sam_neck", False)
            self.config.train_sam_patch_embed = kwargs.get("train_sam_patch_embed", False)
            self.config.train_sam_prompt_encoder = kwargs.get("train_sam_prompt_encoder", False)
            self.config.train_sam_last_blocks = kwargs.get("train_sam_last_blocks", 0)
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.config.train_sam_neck = kwargs.get(
                "train_sam_neck", getattr(self.config, "train_sam_neck", False)
            )
            self.config.train_sam_patch_embed = kwargs.get(
                "train_sam_patch_embed", getattr(self.config, "train_sam_patch_embed", False)
            )
            self.config.train_sam_prompt_encoder = kwargs.get(
                "train_sam_prompt_encoder", getattr(self.config, "train_sam_prompt_encoder", False)
            )
            self.config.train_sam_last_blocks = kwargs.get(
                "train_sam_last_blocks", getattr(self.config, "train_sam_last_blocks", 0)
            )
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        image_encoder = self.visual_model.image_encoder
        if getattr(config, "train_sam_neck", False):
            image_encoder.neck.train()
            for param in image_encoder.neck.parameters():
                param.requires_grad = True

        if getattr(config, "train_sam_patch_embed", False):
            image_encoder.patch_embed.train()
            for param in image_encoder.patch_embed.parameters():
                param.requires_grad = True

        if getattr(config, "train_sam_prompt_encoder", False):
            self.visual_model.prompt_encoder.train()
            for param in self.visual_model.prompt_encoder.parameters():
                param.requires_grad = True

        num_train_blocks = max(0, min(len(image_encoder.blocks), getattr(config, "train_sam_last_blocks", 0)))
        if num_train_blocks > 0:
            for block in image_encoder.blocks[-num_train_blocks:]:
                block.train()
                for param in block.parameters():
                    param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        
        self.boundary_bce_band_width = kwargs.pop(
            "boundary_bce_band_width",
            getattr(config, "boundary_bce_band_width", 0),
        )
        self.boundary_bce_weight = kwargs.pop(
            "boundary_bce_weight",
            getattr(config, "boundary_bce_weight", 1.0),
        )
        self.rail_ego_side_loss_weight = kwargs.pop(
            "rail_ego_side_loss_weight",
            getattr(config, "rail_ego_side_loss_weight", 0.0),
        )
        config.boundary_bce_band_width = self.boundary_bce_band_width
        config.boundary_bce_weight = self.boundary_bce_weight
        config.rail_ego_side_loss_weight = self.rail_ego_side_loss_weight

        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rail_ego_side_head = nn.Linear(config.out_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
    # Only disable gradients when the SAM image encoder is actually frozen.
        train_image_encoder = any(
            p.requires_grad for p in self.model.visual_model.image_encoder.parameters()
        )

        grad_context = torch.enable_grad if train_image_encoder else torch.no_grad

        with grad_context():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        ce_token_weights: torch.FloatTensor = None,
        rail_ego_side_labels: torch.LongTensor = None,
        reason_seg_weight_maps_list: List[torch.FloatTensor] = None,
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        projected_seg_embeddings = last_hidden_state[seg_token_mask]
        pred_embeddings = projected_seg_embeddings
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits
        
        logits = model_output.logits
        label_pad_len = logits.shape[1] - labels.shape[1]
        if label_pad_len > 0:
            pad_labels = torch.full(
                (labels.shape[0], label_pad_len),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            aligned_labels = torch.cat([pad_labels, labels], dim=1)
            if ce_token_weights is None:
                pad_weights = torch.ones(
                    (labels.shape[0], label_pad_len),
                    dtype=logits.dtype,
                    device=logits.device,
                )
                aligned_token_weights = torch.cat(
                    [pad_weights, torch.ones_like(labels, dtype=logits.dtype)], dim=1
                )
            else:
                pad_weights = torch.ones(
                    (ce_token_weights.shape[0], label_pad_len),
                    dtype=ce_token_weights.dtype,
                    device=ce_token_weights.device,
                )
                aligned_token_weights = torch.cat([pad_weights, ce_token_weights], dim=1)
        else:
            aligned_labels = labels
            if ce_token_weights is None:
                aligned_token_weights = torch.ones_like(labels, dtype=logits.dtype)
            else:
                aligned_token_weights = ce_token_weights

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = aligned_labels[..., 1:].contiguous()
        shift_token_weights = aligned_token_weights[..., 1:].to(shift_logits.dtype).contiguous()

        token_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )
        flat_labels = shift_labels.view(-1)
        flat_token_weights = shift_token_weights.view(-1)
        valid_mask = flat_labels.ne(IGNORE_INDEX)
        if valid_mask.any():
            ce_loss = (token_ce[valid_mask] * flat_token_weights[valid_mask]).sum() / flat_token_weights[valid_mask].sum().clamp_min(1.0)
        else:
            ce_loss = token_ce.new_tensor(0.0)

        ce_loss = ce_loss * self.ce_loss_weight
        rail_ego_side_loss = ce_loss.new_tensor(0.0)
        if self.rail_ego_side_loss_weight > 0:
            dummy_seg_embedding = last_hidden_state.new_zeros(
                (1, last_hidden_state.shape[-1])
            )
            rail_ego_side_loss = (
                self.rail_ego_side_head(dummy_seg_embedding).sum() * 0.0
            )
            if rail_ego_side_labels is not None:
                rail_ego_side_labels = rail_ego_side_labels.to(
                    device=seg_token_counts.device,
                    dtype=torch.long,
                )
                seg_labels = torch.repeat_interleave(
                    rail_ego_side_labels,
                    seg_token_counts,
                )
                valid_seg_labels = seg_labels.ne(IGNORE_INDEX)
                if valid_seg_labels.any():
                    rail_ego_side_logits = self.rail_ego_side_head(
                        projected_seg_embeddings[valid_seg_labels]
                    )
                    rail_ego_side_loss = rail_ego_side_loss + F.cross_entropy(
                        rail_ego_side_logits.float(),
                        seg_labels[valid_seg_labels],
                    )

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )

            pixel_weights = None
            if (
                reason_seg_weight_maps_list is not None
                and batch_idx < len(reason_seg_weight_maps_list)
                and reason_seg_weight_maps_list[batch_idx].numel() > 0
            ):
                pixel_weights = reason_seg_weight_maps_list[batch_idx]

            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0], boundary_band_width=self.boundary_bce_band_width, boundary_weight=self.boundary_bce_weight,pixel_weights=pixel_weights,)
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = (
            ce_loss
            + mask_loss
            + self.rail_ego_side_loss_weight * rail_ego_side_loss
        )

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "rail_ego_side_loss": rail_ego_side_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
