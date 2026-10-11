"""Native Qwen3-VL + unchanged SAM, independent of legacy LLaVA imports.

This stage supplies representations/masks, not the training loss or optimizer.
The training entrypoint must add keepalive_loss even on text-only microbatches.
"""

from dataclasses import replace
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen_environment import load_manifest
from utils.qwen_processing import QwenProcessingConfig, register_seg_token
from .qwen_sam_config import QwenSAMConfig, select_sam_state_dict
from .qwen_seg_tokens import install_seg_token_rows, materialized_seg_rows
from .seg_query_bridge import FourQuerySegBridge, group_prompt_embeddings_by_image
from .segment_anything import build_sam_vit_h


def read_sam_checkpoint(path, expected_keys, source_type):
    """Read consolidated torch weights or selectively read SAM from HF shards."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SAM initialization checkpoint not found: {path}")
    if path.is_file() and path.suffix != ".safetensors":
        return select_sam_state_dict(torch.load(path, map_location="cpu", weights_only=True),
                                     expected_keys, source_type)
    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        if not index_path.is_file():
            index_path = path / "pytorch_model.bin.index.json"
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            names = select_sam_state_dict({key: key for key in weight_map}, expected_keys, source_type)
            selected = {}
            for shard in sorted({weight_map[name] for name in names.values()}):
                shard_path = (path / shard).resolve()
                if not shard_path.is_relative_to(path.resolve()):
                    raise ValueError("Checkpoint index points outside its directory.")
                selected.update(_read_selected_tensors(shard_path,
                    {key: name for key, name in names.items() if weight_map[name] == shard}))
            return selected
        for filename in ("model.safetensors", "pytorch_model.bin"):
            if (path / filename).is_file():
                return read_sam_checkpoint(path / filename, expected_keys, source_type)
        raise ValueError("Provide original SAM weights, a consolidated rail tensor file, or a merged HF "
                         "directory; partitioned DeepSpeed checkpoints must first be consolidated.")
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = select_sam_state_dict({key: key for key in handle.keys()}, expected_keys, source_type)
        return {key: handle.get_tensor(name) for key, name in names.items()}


def _read_selected_tensors(path, names):
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return {key: handle.get_tensor(name) for key, name in names.items()}
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key: state[name] for key, name in names.items()}


def validate_sam_shapes(sam, state):
    expected = sam.state_dict()
    if set(state) != set(expected):
        raise ValueError("SAM checkpoint coverage must be complete and exact.")
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape:
            raise ValueError(f"SAM checkpoint has incompatible tensor: {key}")


def seg_event_positions(input_ids, attention_mask, assistant_mask, seg_token_id):
    """Native token positions already include every dynamically expanded image token."""
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or assistant_mask.shape != input_ids.shape:
        raise ValueError("IDs, attention, and assistant masks must share [conversation, token] shape.")
    valid = attention_mask.bool()
    if bool((assistant_mask.bool() & ~valid).any()):
        raise ValueError("Assistant supervision cannot include padding.")
    anchors = input_ids.eq(seg_token_id) & valid & assistant_mask.bool()
    events = anchors.nonzero(as_tuple=False)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
    preceding = positions.masked_fill(~valid, -1).cummax(dim=1).values
    if events.numel():
        rows, columns = events.unbind(1)
        if bool(columns.eq(0).any()):
            raise ValueError("SEG must have a preceding valid predictor token.")
        predictors = preceding[rows, columns - 1]
        if bool(predictors.lt(0).any()):
            raise ValueError("SEG must have a preceding valid predictor token.")
    else:
        predictors = input_ids.new_empty(0)
    return anchors, events, predictors


class QwenSAM(nn.Module):
    def __init__(self, qwen, sam, processor, config=None, processing_config=None,
                 *, _defer_validation=False):
        super().__init__()
        config = config or QwenSAMConfig()
        self.qwen = qwen
        self.sam = sam
        self.processor = processor
        self.processing_config = processing_config or QwenProcessingConfig()
        seg_token_id = register_seg_token(processor.tokenizer)
        if config.seg_token_id is not None and config.seg_token_id != seg_token_id:
            raise ValueError("Saved SEG identity differs from the processor; refusing remapping.")
        qwen.requires_grad_(False)
        capacity = install_seg_token_rows(qwen, processor.tokenizer, seg_token_id)
        if config.qwen_vocab_capacity is not None and capacity != config.qwen_vocab_capacity:
            raise ValueError("Saved vocabulary capacity differs from native Qwen weights.")
        self.config = replace(config, seg_token_id=seg_token_id, qwen_vocab_capacity=capacity)
        hidden_size = qwen.config.text_config.hidden_size
        native_weight = qwen.get_input_embeddings().weight
        self.text_projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                             nn.Linear(hidden_size, 256), nn.Dropout(0.0))
        self.rail_ego_side_head = nn.Linear(256, 2)
        self.seg_query_bridge = None
        if config.seg_prompt_bridge_type == "four_query":
            self.seg_query_bridge = FourQuerySegBridge(hidden_size=hidden_size,
                num_heads=config.seg_query_num_heads, num_hidden_layers=config.seg_query_num_hidden_layers)
        for module in (self.text_projection, self.rail_ego_side_head, self.seg_query_bridge):
            if module is not None:
                module.to(device=native_weight.device, dtype=native_weight.dtype)
        self.configure_sam_trainability()
        if not _defer_validation:
            self.validate_auxiliary_parameters()

    @property
    def native_qwen(self):
        # Stage 4 may wrap Qwen in PEFT; LoRA layers remain active inside the
        # native multimodal backbone. Do not follow PeftModel.model ambiguously.
        getter = getattr(self.qwen, "get_base_model", None)
        return getter() if callable(getter) else self.qwen

    def configure_sam_trainability(self):
        self.sam.requires_grad_(False)
        for enabled, module in ((self.config.train_mask_decoder, self.sam.mask_decoder),
                                (self.config.train_sam_prompt_encoder, self.sam.prompt_encoder),
                                (self.config.train_sam_neck, self.sam.image_encoder.neck),
                                (self.config.train_sam_patch_embed, self.sam.image_encoder.patch_embed)):
            if enabled:
                module.requires_grad_(True)
        count = min(self.config.train_sam_last_blocks, len(self.sam.image_encoder.blocks))
        if count:
            for block in self.sam.image_encoder.blocks[-count:]:
                block.requires_grad_(True)

    def gradient_checkpointing_enable(self):
        # Reentrant checkpointing can detach captured intermediate states and
        # lose gradients where frozen vision prefixes feed trainable blocks.
        self.native_qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.native_qwen.config.use_cache = False

    def validate_auxiliary_parameters(self):
        modules = (self.text_projection, self.rail_ego_side_head,
                   self.native_qwen.get_input_embeddings(), self.native_qwen.get_output_embeddings())
        for module in modules:
            for name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue
                if parameter.is_meta or not bool(torch.isfinite(parameter).all()):
                    raise ValueError(f"Uninitialized/non-finite trainable auxiliary parameter: {name}")
        if self.seg_query_bridge is not None:
            self.seg_query_bridge.validate_parameters()

    @classmethod
    def from_native_pretrained(cls, model_dir, *, sam_init_source, sam_init_path,
                               config=None, processing_config=None, dtype=torch.bfloat16):
        if sam_init_source not in ("pretrained", "rail_checkpoint") or not sam_init_path:
            raise ValueError("A fresh run requires explicit sam_init_source and sam_init_path.")
        model_dir = Path(model_dir)
        manifest = load_manifest(model_dir)
        sam = build_sam_vit_h(checkpoint=None)
        state = read_sam_checkpoint(sam_init_path, sam.state_dict().keys(), sam_init_source)
        validate_sam_shapes(sam, state)
        sam.load_state_dict(state, strict=True)
        del state
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        # All custom modules are created AFTER HF loading leaves its meta-init
        # context. No random/uninitialized custom bridge weights enter training.
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, local_files_only=True,
            dtype=dtype, attn_implementation="sdpa")
        config = replace(config or QwenSAMConfig(), source_qwen_revision=manifest["resolved_revision"],
                         sam_init_source=sam_init_source, sam_init_path=str(sam_init_path))
        return cls(qwen, sam.to(dtype=dtype), processor, config, processing_config)

    def _language_forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        native = self.native_qwen
        # The multimodal output exposes normalized last_hidden_state. The
        # conditional-generation wrapper's captured last layer can be pre-norm.
        output = native.model(input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False,
            output_hidden_states=self.seg_query_bridge is not None, return_dict=True)
        final = output.last_hidden_state
        if final.shape[:2] != input_ids.shape:
            raise ValueError("Qwen hidden positions must match native expanded token IDs exactly.")
        if self.seg_query_bridge is not None:
            states = output.hidden_states
            if states is None or len(states) != native.config.text_config.num_hidden_layers + 1:
                raise ValueError("Qwen did not return embedding plus all configured text layers.")
            states = (*states[:-1], final)
        else:
            states = (final,)
        return native.get_output_embeddings()(final), states

    def _prompts(self, hidden_states, input_ids, attention_mask, assistant_mask):
        anchors, events, predictors = seg_event_positions(
            input_ids, attention_mask, assistant_mask, self.config.seg_token_id)
        final = hidden_states[-1]
        rows = events[:, 0]
        projected_states = final[rows, predictors]
        valid_indices = attention_mask.bool().nonzero(as_tuple=False)
        if not valid_indices.numel():
            raise ValueError("Qwen/SAM requires at least one valid token.")
        row, column = valid_indices[-1].tolist()
        # Always execute one extra projection/head row on every rank.
        projected_all = self.text_projection(torch.cat([projected_states, final[row, column].unsqueeze(0)]))
        projected = projected_all[:-1]
        ego_all = self.rail_ego_side_head(projected_all)
        keepalive = projected_all[-1].sum() * 0.0 + ego_all[-1].sum() * 0.0
        prompts = projected.unsqueeze(1)
        if self.seg_query_bridge is not None:
            residual, bridge_zero = self.seg_query_bridge(hidden_states, anchors, attention_mask.bool(),
                return_keepalive_loss=True, check_finite=self.config.seg_bridge_numerics_debug)
            scale = self.seg_query_bridge.residual_scale().to(residual.dtype)
            prompts = torch.cat([prompts + residual[:, :1] * scale, residual[:, 1:] * scale], dim=1)
            keepalive = keepalive + bridge_zero
        counts = anchors.sum(-1, dtype=torch.long)
        return prompts, ego_all[:-1], counts, events, predictors, keepalive

    def _decode_masks(self, images, grouped_prompts, resize_list, original_size_list):
        dtype = next(self.sam.parameters()).dtype
        embeddings = self.sam.image_encoder(images.to(dtype=dtype))
        masks = []
        keepalive = embeddings.sum() * 0.0
        for index, prompts in enumerate(grouped_prompts):
            count = prompts.shape[0]
            # An extra dummy group ensures decoder participation even with no
            # SEG, and a zero-loss branch exists on every rank, not only some.
            groups = torch.cat([prompts.to(dtype), prompts.new_zeros(1, prompts.shape[1], 256).to(dtype)])
            sparse, dense = self.sam.prompt_encoder(points=None, boxes=None, masks=None, text_embeds=groups)
            low_res, iou_predictions = self.sam.mask_decoder(image_embeddings=embeddings[index:index + 1],
                image_pe=self.sam.prompt_encoder.get_dense_pe().to(dtype),
                sparse_prompt_embeddings=sparse.to(dtype), dense_prompt_embeddings=dense.to(dtype),
                multimask_output=False)
            if tuple(low_res.shape[:2]) != (count + 1, 1):
                raise ValueError("SAM must produce one mask per SEG prompt group, not per query.")
            keepalive = keepalive + low_res[-1].sum() * 0.0 + iou_predictions[-1].sum() * 0.0
            if count:
                masks.append(self.sam.postprocess_masks(low_res[:count], resize_list[index],
                                                       original_size_list[index])[:, 0])
            else:
                masks.append(embeddings.new_empty((0, *original_size_list[index])))
        # Text-only prompting and single-mask decoding leave point/box/mask
        # branches and unused multimask hypernetworks disconnected. Keep every
        # enabled SAM parameter in all distributed backward graphs.
        for parameter in self.sam.parameters():
            if parameter.requires_grad:
                keepalive = keepalive + parameter.reshape(-1)[0] * 0.0
        return masks, keepalive

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, images,
                offset, resize_list, original_size_list, labels=None,
                assistant_token_mask=None, inference=False, **metadata):
        if assistant_token_mask is None:
            if inference or labels is None:
                raise ValueError("Use generate_masks for prompt-only inference; never count prompt SEG tokens.")
            if labels.shape != input_ids.shape:
                raise ValueError("Training labels must align with native input IDs.")
            assistant_token_mask = labels.ne(-100)
        if offset.ndim != 1 or offset.numel() != images.shape[0] + 1 or int(offset[0]) != 0 or \
           int(offset[-1]) != input_ids.shape[0] or bool((offset[1:] <= offset[:-1]).any()):
            raise ValueError("Offsets must map each SAM image to its nonempty conversation rows.")
        if len(resize_list) != images.shape[0] or len(original_size_list) != images.shape[0]:
            raise ValueError("SAM sizes must align with source images.")
        native = self.native_qwen
        if image_grid_thw.shape != (input_ids.shape[0], 3):
            raise ValueError("Each conversation must carry exactly one native Qwen image grid.")
        merge_size = native.config.vision_config.spatial_merge_size
        expected_visual = image_grid_thw.prod(-1) // merge_size**2
        actual_visual = (input_ids.eq(native.config.image_token_id) & attention_mask.bool()).sum(-1)
        if not torch.equal(expected_visual.to(actual_visual.device), actual_visual):
            raise ValueError("Native Qwen visual-token/grid counts disagree.")
        logits, hidden_states = self._language_forward(input_ids, attention_mask, pixel_values, image_grid_thw)
        prompts, ego_logits, counts, events, predictors, keepalive = self._prompts(
            hidden_states, input_ids, attention_mask, assistant_token_mask)
        grouped = group_prompt_embeddings_by_image(prompts, counts, offset)
        predicted_masks, sam_zero = self._decode_masks(images, grouped, resize_list, original_size_list)
        if labels is not None and "masks_list" in metadata:
            if len(metadata["masks_list"]) != len(predicted_masks):
                raise ValueError("Ground-truth mask lists must align with every source image.")
            for predicted, target in zip(predicted_masks, metadata["masks_list"]):
                if predicted.shape != target.shape:
                    raise ValueError("Supervised SEG events do not match the ordered ground-truth masks.")
        return {"logits": logits, "pred_masks": predicted_masks, "prompt_groups": grouped,
                "rail_ego_side_logits": ego_logits, "seg_token_counts": counts,
                "seg_events": events, "seg_predictor_positions": predictors,
                "keepalive_loss": keepalive + sam_zero}

    @torch.no_grad()
    def generate_masks(self, batch, max_new_tokens=256, **generation_kwargs):
        if not batch.get("inference", False):
            raise ValueError("Generation requires prompt-only samples, not teacher-forced target answers.")
        if max_new_tokens <= 0 or batch["input_ids"].shape[1] + max_new_tokens > self.processing_config.max_seq_length:
            raise ValueError("Prompt plus generation budget exceeds the sequence limit; no truncation performed.")
        if generation_kwargs.get("num_return_sequences", 1) != 1 or "max_length" in generation_kwargs:
            raise ValueError("Return one answer per conversation and use max_new_tokens, not max_length.")
        if not bool(batch["attention_mask"][:, -1].all()):
            raise ValueError("Batched decoder-only inference must use left padding.")
        was_training = self.training
        self.eval()
        try:
            generated = self.qwen.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"],
                max_new_tokens=max_new_tokens, **generation_kwargs)
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            prefix_length = batch["input_ids"].shape[1]
            if sequences.shape[0] != batch["input_ids"].shape[0] or not torch.equal(
                sequences[:, :prefix_length], batch["input_ids"]):
                raise ValueError("Generation changed prompt rows/order; mask alignment would be unsafe.")
            if sequences.shape[1] > self.processing_config.max_seq_length:
                raise ValueError("Generated sequence exceeds the configured sequence limit.")
            eos = generation_kwargs.get("eos_token_id", self.native_qwen.generation_config.eos_token_id)
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            generated_mask = torch.zeros_like(sequences, dtype=torch.bool)
            answers = []
            for row, suffix in enumerate(sequences[:, prefix_length:].tolist()):
                length = next((i + 1 for i, token in enumerate(suffix) if token in eos_ids), len(suffix))
                generated_mask[row, prefix_length:prefix_length + length] = True
                answers.append(self.processor.tokenizer.decode(suffix[:length], skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False))
            attention = torch.zeros_like(sequences)
            attention[:, :prefix_length] = batch["attention_mask"]
            attention[generated_mask] = 1
            # A fresh native image+full-text pass recomputes mRoPE and DeepStack
            # positions; never reuse cached generation-step hidden offsets.
            inputs = {key: batch[key] for key in ("pixel_values", "image_grid_thw", "images", "offset",
                                                  "resize_list", "original_size_list")}
            result = self.forward(input_ids=sequences, attention_mask=attention,
                                  assistant_token_mask=generated_mask, **inputs)
            result.update(answers=answers, output_ids=sequences, attention_mask=attention,
                          missing_seg_conversations=result["seg_token_counts"].eq(0).nonzero().flatten().tolist())
            return result
        finally:
            self.train(was_training)

    def auxiliary_state_dict(self):
        return {"sam": self.sam.state_dict(), "text_projection": self.text_projection.state_dict(),
                "rail_ego_side_head": self.rail_ego_side_head.state_dict(),
                "seg_query_bridge": self.seg_query_bridge.state_dict() if self.seg_query_bridge is not None else None}

    def load_auxiliary_state_dict(self, state):
        if set(state) != {"sam", "text_projection", "rail_ego_side_head", "seg_query_bridge"}:
            raise ValueError("Composite checkpoint requires every auxiliary component.")
        modules = {"sam": self.sam, "text_projection": self.text_projection,
                   "rail_ego_side_head": self.rail_ego_side_head, "seg_query_bridge": self.seg_query_bridge}
        # Check every component before loading; no legacy bridge missing-key
        # fallback/reset is allowed when resuming a genuine composite checkpoint.
        for name, module in modules.items():
            if module is None:
                if state[name] is not None:
                    raise ValueError("Checkpoint bridge type differs from composite configuration.")
            elif not isinstance(state[name], dict) or set(state[name]) != set(module.state_dict()):
                raise ValueError(f"Incomplete auxiliary checkpoint component: {name}")
            else:
                expected = module.state_dict()
                for key, tensor in state[name].items():
                    if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape:
                        raise ValueError(f"Incompatible auxiliary tensor: {name}.{key}")
        for name, module in modules.items():
            if module is not None:
                module.load_state_dict(state[name], strict=True)
        self.validate_auxiliary_parameters()

    def save_pretrained(self, directory):
        if callable(getattr(self.qwen, "get_base_model", None)):
            raise ValueError("Merge language LoRA before native export; use trainer checkpoints for unmerged resume.")
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Refusing to overwrite a nonempty composite checkpoint directory.")
        directory.mkdir(parents=True, exist_ok=True)
        with materialized_seg_rows(self.native_qwen) as native:
            native.save_pretrained(directory / "qwen", safe_serialization=True)
        self.processor.save_pretrained(directory / "processor")
        self.processing_config.save_pretrained(directory / "processor")
        torch.save(self.auxiliary_state_dict(), directory / "qwen_sam_aux.pt")
        self.config.save_pretrained(directory)  # completion marker, written last

    @classmethod
    def from_pretrained(cls, directory, dtype=torch.bfloat16):
        directory = Path(directory)
        config = QwenSAMConfig.from_pretrained(directory)
        if config.seg_token_id is None or config.qwen_vocab_capacity is None:
            raise ValueError("Composite checkpoint is missing token/vocabulary identity.")
        processor = AutoProcessor.from_pretrained(directory / "processor", local_files_only=True)
        processing = QwenProcessingConfig.from_pretrained(directory / "processor")
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(directory / "qwen", local_files_only=True,
                                                            dtype=dtype, attn_implementation="sdpa")
        model = cls(qwen, build_sam_vit_h(checkpoint=None).to(dtype=dtype), processor, config, processing)
        model.load_auxiliary_state_dict(torch.load(directory / "qwen_sam_aux.pt", map_location="cpu",
                                                   weights_only=True))
        return model
