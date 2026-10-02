import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


SEG_BRIDGE_SINGLE = "single"
SEG_BRIDGE_FOUR_QUERY = "four_query"
SEG_BRIDGE_TYPES = (SEG_BRIDGE_SINGLE, SEG_BRIDGE_FOUR_QUERY)


def require_finite_tensor(tensor: torch.Tensor, stage: str) -> None:
    """Raise a compact, stage-specific error when a tensor contains NaN/Inf."""
    if tensor.numel() == 0:
        return
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return

    detached = tensor.detach()
    finite_values = detached[finite_mask]
    finite_min = (
        float(finite_values.float().min().item())
        if finite_values.numel() > 0
        else None
    )
    finite_max = (
        float(finite_values.float().max().item())
        if finite_values.numel() > 0
        else None
    )
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available()
        and torch.distributed.is_initialized()
        else 0
    )
    raise FloatingPointError(
        "Non-finite tensor in four-query path: "
        f"stage={stage} rank={rank} shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"nan={int(torch.isnan(detached).sum().item())} "
        f"posinf={int(torch.isposinf(detached).sum().item())} "
        f"neginf={int(torch.isneginf(detached).sum().item())} "
        f"finite_min={finite_min} finite_max={finite_max}"
    )


def finalize_four_query_bridge_load(
    bridge: "FourQuerySegBridge",
    missing_checkpoint_keys: Sequence[str],
) -> str:
    """Finish loading a bridge that may be absent from an older checkpoint.

    ``low_cpu_mem_usage=True`` constructs parameters on the meta device. Missing
    checkpoint tensors are subsequently materialized with ``torch.empty`` and
    the base LLaMA initializer does not know how to initialize every custom
    bridge parameter. Detect the checkpoint case explicitly and initialize the
    complete bridge only when it was genuinely absent.

    Returns a short status string suitable for startup logging.
    """
    expected_keys = set(bridge.state_dict().keys())
    missing_bridge_keys = set()
    marker = "seg_query_bridge."
    for key in missing_checkpoint_keys:
        if marker not in key:
            continue
        local_key = key.split(marker, 1)[1]
        if local_key in expected_keys:
            missing_bridge_keys.add(local_key)

    loaded_bridge_keys = expected_keys - missing_bridge_keys
    if not missing_bridge_keys:
        status = "loaded_from_checkpoint"
    elif not loaded_bridge_keys or loaded_bridge_keys <= {"residual_gate"}:
        # The source checkpoint predates the bridge. Reset after meta tensors
        # have been materialized so no torch.empty storage reaches training.
        bridge.reset_parameters()
        status = "initialized_for_legacy_checkpoint"
    elif missing_bridge_keys == {"residual_gate"}:
        # Compatibility with the first four-query checkpoints, which predated
        # the bounded residual gate. Preserve every trained bridge weight.
        bridge.reset_residual_gate()
        status = "loaded_with_default_residual_gate"
    else:
        missing_list = ", ".join(sorted(missing_bridge_keys))
        raise RuntimeError(
            "The checkpoint contains only part of the four-query bridge. "
            "Refusing to mix trained and newly initialized bridge weights. "
            f"Missing bridge keys: {missing_list}"
        )

    bridge.validate_parameters()
    return status


def build_multimodal_position_masks(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    expanded_length: int,
    seg_token_idx: int,
    image_token_idx: int,
    image_patch_count: int,
) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
    """Map valid input tokens and SEG anchors into LLaVA's expanded sequence.

    LLaVA replaces each image sentinel with ``image_patch_count`` visual tokens.
    The new bridge anchors on the hidden state at the actual [SEG] position.  The
    legacy bridge anchors on the preceding valid state that predicts [SEG].

    Returns:
        expanded_valid_mask: valid multimodal positions, shape [B, S].
        seg_anchor_mask: actual [SEG] positions, shape [B, S].
        seg_predictor_mask: preceding valid positions for [SEG], shape [B, S].
    """
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "input_ids and attention_mask must both have shape [B, T]: "
            f"{tuple(input_ids.shape)} and {tuple(attention_mask.shape)}"
        )
    if expanded_length < 0:
        raise ValueError("expanded_length must be non-negative")
    if image_patch_count <= 0:
        raise ValueError("image_patch_count must be positive")

    device = input_ids.device
    batch_size, input_length = input_ids.shape
    expanded_valid_mask = torch.zeros(
        (batch_size, expanded_length), dtype=torch.bool, device=device
    )
    seg_anchor_mask = torch.zeros_like(expanded_valid_mask)
    seg_predictor_mask = torch.zeros_like(expanded_valid_mask)
    attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    image_mask = input_ids.eq(image_token_idx)
    if bool((image_mask & ~attention_mask).any()):
        raise ValueError("Image sentinel tokens must be valid sequence positions")
    invalid_position_seen = (~attention_mask).cumsum(dim=1) > 0
    if bool((image_mask & invalid_position_seen).any()):
        raise ValueError(
            "Masked positions before an image sentinel are not supported"
        )

    image_count = image_mask.sum(dim=1)
    expanded_row_length = input_length + image_count * (
        image_patch_count - 1
    )
    if bool((expanded_row_length > expanded_length).any()):
        raise ValueError(
            "Expanded tokens exceed the returned hidden-state length: "
            f"max={int(expanded_row_length.max())}, length={expanded_length}"
        )

    token_positions = torch.arange(input_length, device=device).unsqueeze(0)
    images_before = image_mask.cumsum(dim=1) - image_mask.to(torch.long)
    token_positions = token_positions + images_before * (image_patch_count - 1)

    valid_text_events = (attention_mask & ~image_mask).nonzero(as_tuple=False)
    if valid_text_events.numel() > 0:
        valid_text_rows = valid_text_events[:, 0]
        valid_text_token_positions = valid_text_events[:, 1]
        valid_text_expanded_positions = token_positions[
            valid_text_rows,
            valid_text_token_positions,
        ]
        expanded_valid_mask[
            valid_text_rows,
            valid_text_expanded_positions,
        ] = True

    valid_image_events = (attention_mask & image_mask).nonzero(as_tuple=False)
    if valid_image_events.numel() > 0:
        valid_image_rows = valid_image_events[:, 0]
        valid_image_token_positions = valid_image_events[:, 1]
        valid_image_starts = token_positions[
            valid_image_rows,
            valid_image_token_positions,
        ]
        patch_offsets = torch.arange(image_patch_count, device=device)
        valid_image_expanded_positions = (
            valid_image_starts.unsqueeze(1) + patch_offsets.unsqueeze(0)
        )
        expanded_valid_mask[
            valid_image_rows.unsqueeze(1).expand_as(
                valid_image_expanded_positions
            ),
            valid_image_expanded_positions,
        ] = True

    seg_positions = input_ids.eq(seg_token_idx) & attention_mask
    seg_events = seg_positions.nonzero(as_tuple=False)
    if seg_events.numel() > 0:
        event_rows = seg_events[:, 0]
        event_token_positions = seg_events[:, 1]
        event_expanded_positions = token_positions[
            event_rows, event_token_positions
        ]
        if bool((event_expanded_positions >= expanded_length).any()):
            raise ValueError("A [SEG] anchor lies outside the hidden sequence")
        seg_anchor_mask[event_rows, event_expanded_positions] = True
        for event_row, event_position in zip(
            event_rows.tolist(),
            event_expanded_positions.tolist(),
        ):
            preceding_valid = expanded_valid_mask[event_row, :event_position].nonzero(
                as_tuple=False
            )
            if preceding_valid.numel() == 0:
                raise ValueError("[SEG] must have a preceding valid token")
            predictor_position = int(preceding_valid[-1, 0])
            seg_predictor_mask[event_row, predictor_position] = True

    return expanded_valid_mask, seg_anchor_mask, seg_predictor_mask


def group_prompt_embeddings_by_image(
    prompt_embeddings: torch.Tensor,
    seg_token_counts: torch.LongTensor,
    offset: torch.LongTensor,
) -> list:
    """Group row-major SEG prompt groups by their source image."""
    if prompt_embeddings.ndim != 3:
        raise ValueError(
            "prompt_embeddings must have shape [N_SEG, N_QUERY, D], got "
            f"{tuple(prompt_embeddings.shape)}"
        )
    if seg_token_counts.ndim != 1:
        raise ValueError("seg_token_counts must have shape [N_CONVERSATION]")
    if offset.ndim != 1 or offset.numel() < 2:
        raise ValueError("offset must contain at least the start and end index")
    if int(offset[-1]) != seg_token_counts.numel():
        raise ValueError(
            "offset must index conversation rows: "
            f"offset[-1]={int(offset[-1])}, rows={seg_token_counts.numel()}"
        )

    event_offsets = torch.cat(
        [seg_token_counts.new_zeros(1), seg_token_counts.cumsum(dim=0)], dim=0
    )
    if int(event_offsets[-1]) != prompt_embeddings.shape[0]:
        raise ValueError(
            "SEG counts do not match prompt groups: "
            f"counts={int(event_offsets[-1])}, groups={prompt_embeddings.shape[0]}"
        )
    image_event_offsets = event_offsets[offset.to(event_offsets.device)]
    return [
        prompt_embeddings[int(image_event_offsets[i]) : int(image_event_offsets[i + 1])]
        for i in range(image_event_offsets.numel() - 1)
    ]


class FourQuerySegBridge(nn.Module):
    """Compress a causal, multi-layer LLaMA prefix into four SAM prompts."""

    def __init__(
        self,
        hidden_size: int,
        out_dim: int = 256,
        num_queries: int = 4,
        num_heads: int = 8,
        num_hidden_layers: int = 4,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or out_dim <= 0:
            raise ValueError("hidden_size and out_dim must be positive")
        if num_queries != 4:
            raise ValueError(
                "The four-query bridge requires num_queries=4, got "
                f"{num_queries}"
            )
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if out_dim % num_heads != 0:
            raise ValueError(
                f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
            )
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.num_queries = num_queries
        self.num_hidden_layers = num_hidden_layers

        self.learned_queries = nn.Parameter(torch.empty(num_queries, out_dim))
        self.layer_embeddings = nn.Parameter(
            torch.empty(num_hidden_layers, out_dim)
        )
        # The bridge is introduced as a bounded residual over LISA's original
        # prompt. Starting small preserves SAM's pretrained prompt scale while
        # allowing the four queries to grow into distinct corrections.
        # This must be length-one rather than scalar. Transformers' low-CPU
        # loader materializes parameters with ``torch.empty(*param.size())``;
        # a scalar has no size arguments and makes model loading fail.
        self.residual_gate = nn.Parameter(torch.tensor([0.01]))
        self.anchor_projection = nn.Linear(hidden_size, out_dim)
        self.memory_projection = nn.Linear(hidden_size, out_dim)
        self.query_norm = nn.LayerNorm(out_dim)
        self.memory_norm = nn.LayerNorm(out_dim)
        self.cross_attention = nn.MultiheadAttention(
            out_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.post_attention_norm = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, 4 * out_dim),
            nn.GELU(),
            nn.Linear(4 * out_dim, out_dim),
        )
        self.output_norm = nn.LayerNorm(out_dim)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize every bridge parameter, including custom raw tensors."""
        nn.init.normal_(self.learned_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.layer_embeddings, mean=0.0, std=0.02)

        self.residual_gate.fill_(0.01)
        self.anchor_projection.reset_parameters()
        self.memory_projection.reset_parameters()
        self.query_norm.reset_parameters()
        self.memory_norm.reset_parameters()

        # MultiheadAttention owns raw in-projection parameters in addition to
        # its out-projection Linear, so both initialization paths are required.
        self.cross_attention.out_proj.reset_parameters()
        self.cross_attention._reset_parameters()

        self.post_attention_norm.reset_parameters()
        self.ffn[0].reset_parameters()
        self.ffn[2].reset_parameters()
        self.output_norm.reset_parameters()

    @torch.no_grad()
    def reset_residual_gate(self) -> None:
        """Set the compatibility gate without changing trained bridge weights."""
        self.residual_gate.fill_(0.01)

    def validate_parameters(self) -> None:
        """Fail before distributed setup if any bridge parameter is invalid."""
        for name, parameter in self.named_parameters():
            if parameter.is_meta:
                raise RuntimeError(
                    "Four-query bridge parameter remained on the meta device "
                    f"after checkpoint loading: {name}"
                )
            require_finite_tensor(parameter, f"bridge_parameter_{name}")

    def residual_scale(self) -> torch.Tensor:
        """A bounded, trainable scale for the four-query residual."""
        return torch.tanh(self.residual_gate)

    @staticmethod
    def _linear_fp32(layer: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
        """Run a linear layer in FP32 while retaining gradients to its params."""
        bias = layer.bias.float() if layer.bias is not None else None
        return F.linear(inputs.float(), layer.weight.float(), bias)

    @staticmethod
    def _layer_norm_fp32(
        layer: nn.LayerNorm,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        weight = layer.weight.float() if layer.weight is not None else None
        bias = layer.bias.float() if layer.bias is not None else None
        return F.layer_norm(
            inputs.float(),
            layer.normalized_shape,
            weight,
            bias,
            layer.eps,
        )

    def _cross_attention_fp32(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Equivalent dropout-free MHA with projections and softmax in FP32."""
        attention = self.cross_attention
        embed_dim = attention.embed_dim
        projection_weight = attention.in_proj_weight.float()
        projection_bias = (
            attention.in_proj_bias.float()
            if attention.in_proj_bias is not None
            else None
        )

        query_bias = projection_bias[:embed_dim] if projection_bias is not None else None
        key_bias = (
            projection_bias[embed_dim : 2 * embed_dim]
            if projection_bias is not None
            else None
        )
        value_bias = projection_bias[2 * embed_dim :] if projection_bias is not None else None
        projected_queries = F.linear(
            queries.float(), projection_weight[:embed_dim], query_bias
        )
        projected_keys = F.linear(
            memory.float(),
            projection_weight[embed_dim : 2 * embed_dim],
            key_bias,
        )
        projected_values = F.linear(
            memory.float(), projection_weight[2 * embed_dim :], value_bias
        )

        batch_size, query_count, _ = projected_queries.shape
        memory_length = projected_keys.shape[1]
        num_heads = attention.num_heads
        head_dim = embed_dim // num_heads
        projected_queries = projected_queries.view(
            batch_size, query_count, num_heads, head_dim
        ).transpose(1, 2)
        projected_keys = projected_keys.view(
            batch_size, memory_length, num_heads, head_dim
        ).transpose(1, 2)
        projected_values = projected_values.view(
            batch_size, memory_length, num_heads, head_dim
        ).transpose(1, 2)

        attention_scores = torch.matmul(
            projected_queries,
            projected_keys.transpose(-2, -1),
        ) / math.sqrt(head_dim)
        attention_scores = attention_scores.masked_fill(
            memory_padding_mask[:, None, None, :],
            torch.finfo(attention_scores.dtype).min,
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(
            attention_weights,
            projected_values,
        )
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, query_count, embed_dim
        )
        output_bias = (
            attention.out_proj.bias.float()
            if attention.out_proj.bias is not None
            else None
        )
        return F.linear(
            attention_output,
            attention.out_proj.weight.float(),
            output_bias,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Checkpoints created before the stabilizing residual gate remain
        # loadable with strict=True; they start with the conservative default.
        residual_gate_key = prefix + "residual_gate"
        if residual_gate_key not in state_dict:
            state_dict[residual_gate_key] = self.residual_gate.detach().clone()
        elif state_dict[residual_gate_key].ndim == 0:
            # Retain compatibility with the short-lived scalar-gate revision.
            state_dict[residual_gate_key] = state_dict[residual_gate_key].reshape(1)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def selected_layer_indices(self, hidden_state_count: int) -> Sequence[int]:
        """Select evenly spaced transformer outputs, including the final layer.

        Hugging Face hidden-state tuples contain the embedding output at index 0,
        followed by one output per transformer layer.  The embedding output is
        intentionally excluded.
        """
        transformer_layer_count = hidden_state_count - 1
        if transformer_layer_count <= 0:
            raise ValueError(
                "The four-query bridge requires transformer-layer hidden states"
            )
        selected_count = min(
            self.num_hidden_layers,
            transformer_layer_count,
        )
        return [
            max(
                1,
                math.ceil(
                    transformer_layer_count * (slot + 1) / selected_count
                ),
            )
            for slot in range(selected_count)
        ]

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        seg_anchor_mask: torch.BoolTensor,
        expanded_valid_mask: torch.BoolTensor,
        return_keepalive_loss: bool = False,
        check_finite: bool = False,
    ):
        """Return one four-query prompt group for every [SEG] event.

        Args:
            hidden_states: embedding plus transformer-layer states, each [B,S,H].
            seg_anchor_mask: actual expanded [SEG] positions, [B,S].
            expanded_valid_mask: valid expanded multimodal positions, [B,S].

        Returns:
            By default, a tensor with shape [N_SEG, 4, 256]. If
            ``return_keepalive_loss`` is true, also returns a scalar zero-loss
            term that keeps every bridge parameter in the distributed backward
            graph when this rank has no [SEG] event.
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise TypeError(
                "hidden_states must be a tuple/list containing several layers"
            )
        if len(hidden_states) < 2:
            raise ValueError("At least one transformer hidden state is required")

        final_hidden_state = hidden_states[-1]
        expected_prefix_shape = final_hidden_state.shape[:2]
        if seg_anchor_mask.shape != expected_prefix_shape:
            raise ValueError(
                "seg_anchor_mask must match hidden-state batch/sequence shape: "
                f"{tuple(seg_anchor_mask.shape)} != {tuple(expected_prefix_shape)}"
            )
        if expanded_valid_mask.shape != expected_prefix_shape:
            raise ValueError(
                "expanded_valid_mask must match hidden-state batch/sequence shape: "
                f"{tuple(expanded_valid_mask.shape)} != {tuple(expected_prefix_shape)}"
            )
        for hidden_state in hidden_states:
            if hidden_state.shape[:2] != expected_prefix_shape:
                raise ValueError("All hidden-state layers must share [B,S]")
            if hidden_state.shape[-1] != self.hidden_size:
                raise ValueError(
                    "Unexpected hidden size: "
                    f"{hidden_state.shape[-1]} != {self.hidden_size}"
                )

        seg_anchor_mask = seg_anchor_mask.to(
            device=final_hidden_state.device,
            dtype=torch.bool,
        )
        valid_mask = expanded_valid_mask.to(
            device=final_hidden_state.device,
            dtype=torch.bool,
        )
        if bool((seg_anchor_mask & ~valid_mask).any()):
            raise ValueError("Every [SEG] anchor must be a valid sequence position")

        anchor_indices = seg_anchor_mask.nonzero(as_tuple=False)
        num_seg_events = anchor_indices.shape[0]

        selected_indices = self.selected_layer_indices(len(hidden_states))
        sequence_positions = torch.arange(
            expected_prefix_shape[1],
            device=final_hidden_state.device,
        )
        if num_seg_events > 0:
            dummy_anchor_index = anchor_indices[:1]
        else:
            valid_positions = valid_mask.nonzero(as_tuple=False)
            if valid_positions.numel() == 0:
                raise ValueError("The bridge requires at least one valid token")
            dummy_anchor_index = valid_positions[-1:].clone()
        # Every rank executes the entire bridge at least once, even if its local
        # micro-batch has no [SEG]. This keeps ZeRO's gradient-hook sequence
        # identical across ranks while the appended prompt has zero loss weight.
        all_anchor_indices = torch.cat(
            [anchor_indices, dummy_anchor_index],
            dim=0,
        )
        memory_per_event = []
        anchor_states = []
        for batch_idx, anchor_position in all_anchor_indices.tolist():
            causal_positions = sequence_positions <= anchor_position
            causal_valid = valid_mask[batch_idx] & causal_positions
            if not bool(causal_valid.any()):
                raise ValueError("Every [SEG] anchor must have a valid causal prefix")
            event_memory = []
            for layer_slot, layer_idx in enumerate(selected_indices):
                causal_hidden_states = hidden_states[layer_idx][
                    batch_idx, causal_valid
                ]
                if check_finite:
                    require_finite_tensor(
                        causal_hidden_states,
                        f"llama_layer_{layer_idx}_causal_states",
                    )
                event_memory.append(
                    self._linear_fp32(
                        self.memory_projection,
                        causal_hidden_states,
                    )
                    + self.layer_embeddings[layer_slot].float().unsqueeze(0)
                )
            memory_per_event.append(torch.cat(event_memory, dim=0))
            anchor_states.append(
                final_hidden_state[batch_idx, anchor_position]
            )

        memory_lengths = torch.tensor(
            [memory.shape[0] for memory in memory_per_event],
            device=final_hidden_state.device,
        )
        padded_memory = pad_sequence(memory_per_event, batch_first=True)
        memory_padding_mask = torch.arange(
            padded_memory.shape[1], device=final_hidden_state.device
        ).unsqueeze(0) >= memory_lengths.unsqueeze(1)

        anchor_states = torch.stack(anchor_states, dim=0)
        if check_finite:
            require_finite_tensor(anchor_states, "seg_anchor_states")
            require_finite_tensor(padded_memory, "projected_causal_memory_fp32")
        conditioned_queries = self.learned_queries.float().unsqueeze(0) + (
            self._linear_fp32(
                self.anchor_projection,
                anchor_states,
            ).unsqueeze(1)
        )
        normalized_memory = self._layer_norm_fp32(
            self.memory_norm,
            padded_memory,
        )
        normalized_queries = self._layer_norm_fp32(
            self.query_norm,
            conditioned_queries,
        )
        attention_output = self._cross_attention_fp32(
            normalized_queries,
            normalized_memory,
            memory_padding_mask,
        )
        queries = conditioned_queries + attention_output
        normalized_queries = self._layer_norm_fp32(
            self.post_attention_norm,
            queries,
        )
        ffn_hidden = self._linear_fp32(self.ffn[0], normalized_queries)
        ffn_hidden = F.gelu(ffn_hidden, approximate=self.ffn[1].approximate)
        queries = queries + self._linear_fp32(self.ffn[2], ffn_hidden)
        queries = self._layer_norm_fp32(self.output_norm, queries)
        if check_finite:
            require_finite_tensor(conditioned_queries, "conditioned_queries_fp32")
            require_finite_tensor(normalized_memory, "normalized_memory_fp32")
            require_finite_tensor(attention_output, "cross_attention_output_fp32")
            require_finite_tensor(queries, "bridge_output_fp32")
        prompt_groups = queries[:num_seg_events].to(final_hidden_state.dtype)
        keepalive_loss = (
            queries[num_seg_events:].sum() * 0.0
            + self.residual_scale().float().sum() * 0.0
        ).to(final_hidden_state.dtype)
        if check_finite:
            require_finite_tensor(prompt_groups, "bridge_output_cast")
        if return_keepalive_loss:
            return prompt_groups, keepalive_loss
        return prompt_groups
