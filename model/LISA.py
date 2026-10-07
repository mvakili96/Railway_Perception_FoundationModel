import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)
from model.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

from .seg_query_bridge import (
    SEG_BRIDGE_FOUR_QUERY,
    SEG_BRIDGE_SINGLE,
    SEG_BRIDGE_TYPES,
    FourQuerySegBridge,
    build_multimodal_position_masks,
    finalize_four_query_bridge_load,
    group_prompt_embeddings_by_image,
    require_finite_tensor,
)

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
        self.config.seg_prompt_bridge_type = kwargs.get(
            "seg_prompt_bridge_type",
            getattr(self.config, "seg_prompt_bridge_type", SEG_BRIDGE_SINGLE),
        )
        if self.config.seg_prompt_bridge_type not in SEG_BRIDGE_TYPES:
            raise ValueError(
                "seg_prompt_bridge_type must be one of "
                f"{SEG_BRIDGE_TYPES}, got {self.config.seg_prompt_bridge_type!r}"
            )
        self.config.seg_query_num_queries = kwargs.get(
            "seg_query_num_queries",
            getattr(self.config, "seg_query_num_queries", 4),
        )
        self.config.seg_query_num_heads = kwargs.get(
            "seg_query_num_heads",
            getattr(self.config, "seg_query_num_heads", 8),
        )
        self.config.seg_query_num_hidden_layers = kwargs.get(
            "seg_query_num_hidden_layers",
            getattr(self.config, "seg_query_num_hidden_layers", 4),
        )
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

        if config.seg_prompt_bridge_type == SEG_BRIDGE_FOUR_QUERY:
            self.seg_query_bridge = FourQuerySegBridge(
                hidden_size=in_dim,
                out_dim=out_dim,
                num_queries=config.seg_query_num_queries,
                num_heads=config.seg_query_num_heads,
                num_hidden_layers=config.seg_query_num_hidden_layers,
            )
            config.seg_query_layer_indices = list(
                self.seg_query_bridge.selected_layer_indices(
                    config.num_hidden_layers + 1
                )
            )
            self.seg_query_bridge.train()
            for param in self.seg_query_bridge.parameters():
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

        config.seg_prompt_bridge_type = kwargs.pop(
            "seg_prompt_bridge_type",
            getattr(config, "seg_prompt_bridge_type", SEG_BRIDGE_SINGLE),
        )
        if config.seg_prompt_bridge_type not in SEG_BRIDGE_TYPES:
            raise ValueError(
                "seg_prompt_bridge_type must be one of "
                f"{SEG_BRIDGE_TYPES}, got {config.seg_prompt_bridge_type!r}"
            )
        config.seg_query_num_queries = kwargs.pop(
            "seg_query_num_queries",
            getattr(config, "seg_query_num_queries", 4),
        )
        config.seg_query_num_heads = kwargs.pop(
            "seg_query_num_heads",
            getattr(config, "seg_query_num_heads", 8),
        )
        config.seg_query_num_hidden_layers = kwargs.pop(
            "seg_query_num_hidden_layers",
            getattr(config, "seg_query_num_hidden_layers", 4),
        )

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
        # Runtime-only diagnostic switch. It is intentionally not persisted in
        # the model config so production inference is not forced to synchronize
        # the GPU for numerical checks.
        self.seg_bridge_numerics_debug = False

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rail_ego_side_head = nn.Linear(config.out_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, *model_args, **kwargs):
        """Load LISA and safely initialize a bridge absent from old weights."""
        return_loading_info = kwargs.pop("output_loading_info", False)
        model, loading_info = super().from_pretrained(
            *model_args,
            output_loading_info=True,
            **kwargs,
        )
        model._seg_query_bridge_load_status = "not_enabled"
        if model._uses_four_query_bridge():
            model._seg_query_bridge_load_status = finalize_four_query_bridge_load(
                model.get_model().seg_query_bridge,
                loading_info.get("missing_keys", ()),
            )
        if return_loading_info:
            return model, loading_info
        return model

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

    def _uses_four_query_bridge(self) -> bool:
        return (
            self.config.seg_prompt_bridge_type == SEG_BRIDGE_FOUR_QUERY
        )

    def _get_image_patch_count(self) -> int:
        vision_tower = self.get_vision_tower()
        if vision_tower is None or not hasattr(vision_tower, "num_patches"):
            raise ValueError(
                "The segmentation bridge requires a vision tower exposing "
                "num_patches"
            )
        return int(vision_tower.num_patches)

    def _forward_multimodal_hidden_states(
        self,
        input_ids: torch.LongTensor,
        attention_masks: torch.Tensor,
        images_clip: torch.FloatTensor,
    ):
        """Return all LLaMA layers from a full, no-cache multimodal pass."""
        dummy_labels = torch.full_like(input_ids, IGNORE_INDEX)
        (
            prepared_input_ids,
            expanded_attention_mask,
            _,
            inputs_embeds,
            _,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_masks,
            None,
            dummy_labels,
            images_clip,
        )
        outputs = self.model(
            input_ids=prepared_input_ids,
            attention_mask=expanded_attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states

    def _build_seg_prompt_groups(
        self,
        hidden_states,
        input_ids: torch.LongTensor,
        attention_masks: torch.Tensor,
    ):
        """Build SAM prompt groups and the legacy auxiliary SEG embeddings."""
        if isinstance(hidden_states, (tuple, list)):
            final_hidden_state = hidden_states[-1]
        else:
            final_hidden_state = hidden_states

        (
            expanded_valid_mask,
            seg_anchor_mask,
            seg_predictor_mask,
        ) = build_multimodal_position_masks(
            input_ids=input_ids,
            attention_mask=attention_masks,
            expanded_length=final_hidden_state.shape[1],
            seg_token_idx=self.seg_token_idx,
            image_token_idx=IMAGE_TOKEN_INDEX,
            image_patch_count=self._get_image_patch_count(),
        )

        seg_token_counts = seg_predictor_mask.sum(dim=1, dtype=torch.long)
        anchor_counts = seg_anchor_mask.sum(dim=1, dtype=torch.long)
        if not torch.equal(seg_token_counts, anchor_counts):
            raise ValueError(
                "Every [SEG] token must have exactly one predictor and anchor"
            )

        # Keep the original pre-[SEG] projection intact for the legacy bridge
        # and the existing auxiliary ego-side loss.
        projected_seg_embeddings = self.model.text_hidden_fcs[0](
            final_hidden_state[seg_predictor_mask]
        )
        if self._uses_four_query_bridge():
            check_finite = bool(self.seg_bridge_numerics_debug)
            if check_finite:
                require_finite_tensor(
                    projected_seg_embeddings,
                    "legacy_projected_seg_embeddings",
                )
            if not isinstance(hidden_states, (tuple, list)):
                raise ValueError(
                    "The four-query bridge requires hidden states from several "
                    "LLaMA layers"
                )
            bridge_prompt_groups, bridge_keepalive_loss = (
                self.model.seg_query_bridge(
                    hidden_states,
                    seg_anchor_mask,
                    expanded_valid_mask,
                    return_keepalive_loss=True,
                    check_finite=check_finite,
                )
            )
            bridge_scale = self.model.seg_query_bridge.residual_scale().to(
                dtype=bridge_prompt_groups.dtype
            )
            if check_finite:
                require_finite_tensor(bridge_scale, "bridge_residual_scale")
            # Query 0 starts from the original, already calibrated LISA prompt.
            # The remaining queries begin as small learned residual prompts.
            prompt_groups = torch.cat(
                [
                    projected_seg_embeddings.unsqueeze(1)
                    + bridge_prompt_groups[:, :1] * bridge_scale,
                    bridge_prompt_groups[:, 1:] * bridge_scale,
                ],
                dim=1,
            )
            if check_finite:
                require_finite_tensor(prompt_groups, "gated_sam_prompt_groups")
        else:
            prompt_groups = projected_seg_embeddings.unsqueeze(1)
            bridge_keepalive_loss = final_hidden_state.new_zeros(())

        expected_query_count = (
            self.config.seg_query_num_queries
            if self._uses_four_query_bridge()
            else 1
        )
        expected_prompt_shape = (
            int(seg_token_counts.sum()),
            expected_query_count,
            self.config.out_dim,
        )
        if tuple(prompt_groups.shape) != expected_prompt_shape:
            raise ValueError(
                "Unexpected language-to-SAM prompt shape: "
                f"{tuple(prompt_groups.shape)} != {expected_prompt_shape}"
            )
        return (
            prompt_groups,
            projected_seg_embeddings,
            seg_token_counts,
            bridge_keepalive_loss,
        )

    def _decode_seg_prompt_groups(
        self,
        image_embeddings: torch.Tensor,
        prompt_groups,
        resize_list,
        original_size_list,
        diagnostic_image_paths=None,
    ):
        """Decode one mask for each sparse-prompt group."""
        pred_masks = []
        sam_keepalive_loss = image_embeddings.new_zeros(())
        check_finite = bool(
            self._uses_four_query_bridge()
            and self.seg_bridge_numerics_debug
        )
        if check_finite:
            require_finite_tensor(image_embeddings, "sam_image_embeddings")
        for image_idx, image_prompt_groups in enumerate(prompt_groups):
            image_label = str(image_idx)
            if (
                diagnostic_image_paths is not None
                and image_idx < len(diagnostic_image_paths)
            ):
                image_label = (
                    f"{image_idx}_{os.path.basename(diagnostic_image_paths[image_idx])}"
                )
            original_size = tuple(original_size_list[image_idx])
            num_real_prompt_groups = image_prompt_groups.shape[0]
            if check_finite:
                require_finite_tensor(
                    image_prompt_groups,
                    f"sam_prompt_groups_image_{image_label}",
                )
            if num_real_prompt_groups == 0:
                decoder_prompt_groups = image_prompt_groups.new_zeros(
                    (1, *image_prompt_groups.shape[1:])
                )
            else:
                decoder_prompt_groups = image_prompt_groups

            sparse_embeddings, dense_embeddings = (
                self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=decoder_prompt_groups,
                )
            )
            sparse_embeddings = sparse_embeddings.to(decoder_prompt_groups.dtype)
            if check_finite:
                require_finite_tensor(
                    sparse_embeddings,
                    f"sam_sparse_embeddings_image_{image_label}",
                )
                require_finite_tensor(
                    dense_embeddings,
                    f"sam_dense_embeddings_image_{image_label}",
                )
            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[image_idx].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            if check_finite:
                require_finite_tensor(
                    low_res_masks,
                    f"sam_low_res_masks_image_{image_label}",
                )
            if low_res_masks.shape[:2] != (
                decoder_prompt_groups.shape[0],
                1,
            ):
                raise ValueError(
                    "SAM must return one mask per [SEG] prompt group, got "
                    f"{tuple(low_res_masks.shape[:2])} for "
                    f"{num_real_prompt_groups} real groups"
                )
            sam_keepalive_loss = sam_keepalive_loss + (
                low_res_masks[num_real_prompt_groups:].sum() * 0.0
            )
            if num_real_prompt_groups == 0:
                pred_masks.append(
                    image_embeddings.new_empty((0, *original_size))
                )
                continue
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks[:num_real_prompt_groups],
                input_size=resize_list[image_idx],
                original_size=original_size,
            )
            if check_finite:
                require_finite_tensor(
                    pred_mask,
                    f"sam_postprocessed_masks_image_{image_label}",
                )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks, sam_keepalive_loss

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
        rail_switch_token_masks: torch.BoolTensor = None,
        rail_right_state_token_masks: torch.BoolTensor = None,
        rail_ego_side_labels: torch.LongTensor = None,
        reason_seg_weight_maps_list: List[torch.FloatTensor] = None,
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        if inference:
            images_clip_list = []
            for image_idx in range(len(offset) - 1):
                start_i, end_i = offset[image_idx], offset[image_idx + 1]
                images_clip_list.append(
                    images_clip[image_idx]
                    .unsqueeze(0)
                    .expand(int(end_i - start_i), -1, -1, -1)
                    .contiguous()
                )
            images_clip_expanded = torch.cat(images_clip_list, dim=0)
            output_hidden_states = self._forward_multimodal_hidden_states(
                input_ids,
                attention_masks,
                images_clip_expanded,
            )
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

        assert len(self.model.text_hidden_fcs) == 1
        (
            prompt_groups,
            projected_seg_embeddings,
            seg_token_counts,
            bridge_keepalive_loss,
        ) = self._build_seg_prompt_groups(
            output_hidden_states,
            input_ids,
            attention_masks,
        )
        prompt_groups = group_prompt_embeddings_by_image(
            prompt_groups,
            seg_token_counts,
            offset,
        )
        pred_masks, sam_keepalive_loss = self._decode_seg_prompt_groups(
            image_embeddings,
            prompt_groups,
            resize_list,
            [label.shape for label in label_list],
            diagnostic_image_paths=kwargs.get("image_paths"),
        )

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits
        
        logits = model_output.logits
        if rail_switch_token_masks is None:
            rail_switch_token_masks = torch.zeros_like(labels, dtype=torch.bool)
        elif rail_switch_token_masks.shape != labels.shape:
            raise ValueError(
                "rail_switch_token_masks must have the same shape as labels: "
                f"{rail_switch_token_masks.shape} != {labels.shape}"
            )
        rail_switch_token_masks = rail_switch_token_masks.to(
            device=labels.device,
            dtype=torch.bool,
        )
        if rail_right_state_token_masks is None:
            rail_right_state_token_masks = torch.zeros_like(
                labels,
                dtype=torch.bool,
            )
        elif rail_right_state_token_masks.shape != labels.shape:
            raise ValueError(
                "rail_right_state_token_masks must have the same shape as labels: "
                f"{rail_right_state_token_masks.shape} != {labels.shape}"
            )
        rail_right_state_token_masks = rail_right_state_token_masks.to(
            device=labels.device,
            dtype=torch.bool,
        )

        label_pad_len = logits.shape[1] - labels.shape[1]
        if label_pad_len > 0:
            pad_labels = torch.full(
                (labels.shape[0], label_pad_len),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            aligned_labels = torch.cat([pad_labels, labels], dim=1)
            pad_switch_masks = torch.zeros(
                (rail_switch_token_masks.shape[0], label_pad_len),
                dtype=torch.bool,
                device=rail_switch_token_masks.device,
            )
            aligned_switch_token_masks = torch.cat(
                [pad_switch_masks, rail_switch_token_masks], dim=1
            )
            pad_right_state_masks = torch.zeros(
                (rail_right_state_token_masks.shape[0], label_pad_len),
                dtype=torch.bool,
                device=rail_right_state_token_masks.device,
            )
            aligned_right_state_token_masks = torch.cat(
                [pad_right_state_masks, rail_right_state_token_masks], dim=1
            )
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
            aligned_switch_token_masks = rail_switch_token_masks
            aligned_right_state_token_masks = rail_right_state_token_masks
            if ce_token_weights is None:
                aligned_token_weights = torch.ones_like(labels, dtype=logits.dtype)
            else:
                aligned_token_weights = ce_token_weights

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = aligned_labels[..., 1:].contiguous()
        shift_token_weights = aligned_token_weights[..., 1:].to(shift_logits.dtype).contiguous()
        shift_switch_token_masks = aligned_switch_token_masks[..., 1:].contiguous()
        shift_right_state_token_masks = (
            aligned_right_state_token_masks[..., 1:].contiguous()
        )

        token_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )
        flat_labels = shift_labels.view(-1)
        flat_token_weights = shift_token_weights.view(-1)
        valid_mask = flat_labels.ne(IGNORE_INDEX)
        flat_switch_token_masks = shift_switch_token_masks.view(-1)
        valid_switch_mask = valid_mask & flat_switch_token_masks
        flat_right_state_token_masks = shift_right_state_token_masks.view(-1)
        valid_right_state_mask = valid_mask & flat_right_state_token_masks
        if valid_mask.any():
            ce_loss = (token_ce[valid_mask] * flat_token_weights[valid_mask]).sum() / flat_token_weights[valid_mask].sum().clamp_min(1.0)
        else:
            ce_loss = token_ce.new_tensor(0.0)

        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        switch_token_count = valid_switch_mask.sum()
        if valid_switch_mask.any():
            switch_ce = token_ce[valid_switch_mask].mean()
            switch_predictions = flat_shift_logits[
                valid_switch_mask
            ].argmax(dim=-1)
            switch_targets = flat_labels[valid_switch_mask]
            switch_correct_count = switch_predictions.eq(switch_targets).sum()
            switch_accuracy = (
                switch_correct_count.to(switch_ce.dtype)
                / switch_token_count.to(switch_ce.dtype)
            )
        else:
            switch_ce = token_ce.new_tensor(0.0)
            switch_correct_count = switch_token_count.new_zeros(())
            switch_accuracy = token_ce.new_tensor(0.0)

        right_state_token_count = valid_right_state_mask.sum()
        if valid_right_state_mask.any():
            right_state_ce = token_ce[valid_right_state_mask].mean()
            right_state_predictions = flat_shift_logits[
                valid_right_state_mask
            ].argmax(dim=-1)
            right_state_targets = flat_labels[valid_right_state_mask]
            right_state_correct_count = right_state_predictions.eq(
                right_state_targets
            ).sum()
            right_state_accuracy = (
                right_state_correct_count.to(right_state_ce.dtype)
                / right_state_token_count.to(right_state_ce.dtype)
            )
        else:
            right_state_ce = token_ce.new_tensor(0.0)
            right_state_correct_count = right_state_token_count.new_zeros(())
            right_state_accuracy = token_ce.new_tensor(0.0)

        ce_loss = ce_loss * self.ce_loss_weight
        rail_ego_side_loss = ce_loss.new_tensor(0.0)
        if self.rail_ego_side_loss_weight > 0:
            dummy_seg_embedding = projected_seg_embeddings.new_zeros(
                (1, self.config.out_dim)
            )
            # Make exactly one ego-side-head call on every rank. A semantic
            # micro-batch may have no valid ego-side labels while another rank
            # has reasoning samples; calling the head once with a zero-weight
            # dummy row keeps DeepSpeed ZeRO's gradient hooks synchronized.
            side_head_embeddings = torch.cat(
                [projected_seg_embeddings, dummy_seg_embedding],
                dim=0,
            )
            side_head_logits = self.rail_ego_side_head(side_head_embeddings)
            rail_ego_side_loss = side_head_logits[-1].sum() * 0.0
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
                    rail_ego_side_loss = rail_ego_side_loss + F.cross_entropy(
                        side_head_logits[:-1][valid_seg_labels].float(),
                        seg_labels[valid_seg_labels],
                    )

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            image_label = str(batch_idx)
            diagnostic_image_paths = kwargs.get("image_paths")
            if (
                diagnostic_image_paths is not None
                and batch_idx < len(diagnostic_image_paths)
            ):
                image_label = (
                    f"{batch_idx}_{os.path.basename(diagnostic_image_paths[batch_idx])}"
                )

            if self._uses_four_query_bridge() and self.seg_bridge_numerics_debug:
                require_finite_tensor(
                    gt_mask,
                    f"ground_truth_masks_image_{image_label}",
                )
                require_finite_tensor(
                    pred_mask,
                    f"mask_loss_predictions_image_{image_label}",
                )

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
                if (
                    self._uses_four_query_bridge()
                    and self.seg_bridge_numerics_debug
                ):
                    require_finite_tensor(
                        pixel_weights,
                        f"mask_pixel_weights_image_{image_label}",
                    )

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
            + bridge_keepalive_loss
            + sam_keepalive_loss
        )

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "switch_ce": switch_ce,
            "switch_token_count": switch_token_count,
            "switch_correct_count": switch_correct_count,
            "switch_accuracy": switch_accuracy,
            "right_state_ce": right_state_ce,
            "right_state_token_count": right_state_token_count,
            "right_state_correct_count": right_state_correct_count,
            "right_state_accuracy": right_state_accuracy,
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
        do_sample=None,
    ):
        with torch.no_grad():
            pad_token_id = self.config.pad_token_id
            if pad_token_id is None:
                input_attention_masks = torch.ones_like(
                    input_ids,
                    dtype=torch.bool,
                )
            else:
                input_attention_masks = input_ids.ne(pad_token_id)
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                attention_mask=input_attention_masks,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                return_dict_in_generate=True,
                **({"do_sample": do_sample} if do_sample is not None else {}),
            )
            output_ids = outputs.sequences

            generated_token_count = output_ids.shape[1] - input_ids.shape[1]
            generated_attention_masks = torch.cat(
                [
                    input_attention_masks,
                    torch.ones(
                        (output_ids.shape[0], generated_token_count),
                        dtype=torch.bool,
                        device=output_ids.device,
                    ),
                ],
                dim=1,
            )
            output_hidden_states = self._forward_multimodal_hidden_states(
                output_ids,
                generated_attention_masks,
                images_clip,
            )
            prompt_groups, _, seg_token_counts, _ = (
                self._build_seg_prompt_groups(
                    output_hidden_states,
                    output_ids,
                    generated_attention_masks,
                )
            )
            batch_offset = torch.arange(
                output_ids.shape[0] + 1,
                dtype=torch.long,
                device=output_ids.device,
            )
            prompt_groups = group_prompt_embeddings_by_image(
                prompt_groups,
                seg_token_counts,
                batch_offset,
            )

            image_embeddings = self.get_visual_embs(images)
            pred_masks, _ = self._decode_seg_prompt_groups(
                image_embeddings,
                prompt_groups,
                resize_list,
                original_size_list,
            )

        return output_ids, pred_masks
