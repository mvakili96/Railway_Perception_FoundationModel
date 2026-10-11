"""Learn only SEG vocabulary entries while keeping native Qwen matrices frozen."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class SegTokenEmbedding(nn.Module):
    def __init__(self, base, seg_token_id):
        super().__init__()
        self.base = base
        self.seg_token_id = seg_token_id
        self.base.requires_grad_(False)
        self.seg_row = nn.Parameter(base.weight[seg_token_id].detach().clone())

    @property
    def weight(self):
        return self.base.weight

    @property
    def num_embeddings(self):
        return self.base.num_embeddings

    @property
    def embedding_dim(self):
        return self.base.embedding_dim

    @property
    def padding_idx(self):
        return self.base.padding_idx

    def forward(self, input_ids):
        original = self.base(input_ids)
        return torch.where(input_ids.eq(self.seg_token_id).unsqueeze(-1),
                           self.seg_row.to(original.dtype), original)


class SegTokenOutput(nn.Module):
    def __init__(self, base, seg_token_id):
        super().__init__()
        if base.bias is not None:
            raise ValueError("Native Qwen's output projection must be bias-free.")
        self.base = base
        self.seg_token_id = seg_token_id
        self.base.requires_grad_(False)
        self.seg_row = nn.Parameter(base.weight[seg_token_id].detach().clone())

    @property
    def weight(self):
        return self.base.weight

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, hidden_states):
        logits = self.base(hidden_states)
        # CopySlices preserves hidden-state gradients and avoids a second full
        # vocabulary-sized logits allocation. The native linear saves its input,
        # not these output logits, for backward.
        logits[..., self.seg_token_id] = F.linear(
            hidden_states, self.seg_row.to(hidden_states.dtype).unsqueeze(0),
        ).squeeze(-1)
        return logits


def install_seg_token_rows(qwen, tokenizer, seg_token_id):
    if isinstance(qwen.get_input_embeddings(), SegTokenEmbedding):
        raise ValueError("SEG rows are already installed.")
    capacity = qwen.get_input_embeddings().weight.shape[0]
    # Qwen's padded vocabulary is usually larger than len(tokenizer). Never
    # resize down to the tokenizer length and discard pretrained native rows.
    required = max(capacity, len(tokenizer), seg_token_id + 1)
    if required > capacity:
        qwen.resize_token_embeddings(required, mean_resizing=False)
    input_base = qwen.get_input_embeddings()
    output_base = qwen.get_output_embeddings()
    if input_base.weight.shape != output_base.weight.shape:
        raise ValueError("Qwen input/output vocabulary dimensions disagree.")
    if input_base.weight is output_base.weight or (not input_base.weight.is_meta and
            input_base.weight.data_ptr() == output_base.weight.data_ptr()):
        raise ValueError("Separate SEG entries require untied Qwen input/output weights.")
    qwen.set_input_embeddings(SegTokenEmbedding(input_base, seg_token_id))
    qwen.set_output_embeddings(SegTokenOutput(output_base, seg_token_id))
    return required


@contextmanager
def materialized_seg_rows(qwen):
    """Temporarily expose native weights for export; restore live training state."""
    embedding = qwen.get_input_embeddings()
    output = qwen.get_output_embeddings()
    if not isinstance(embedding, SegTokenEmbedding) or not isinstance(output, SegTokenOutput):
        raise ValueError("Expected installed trainable SEG rows before native export.")
    if embedding.seg_token_id != output.seg_token_id:
        raise ValueError("Input/output SEG identities differ.")
    if embedding.weight.data_ptr() == output.weight.data_ptr():
        raise ValueError("Separate trainable SEG rows require untied native vocabulary weights.")
    index = embedding.seg_token_id
    input_old = embedding.weight[index].detach().clone()
    output_old = output.weight[index].detach().clone()
    try:
        with torch.no_grad():
            embedding.weight[index].copy_(embedding.seg_row)
            output.weight[index].copy_(output.seg_row)
        qwen.set_input_embeddings(embedding.base)
        qwen.set_output_embeddings(output.base)
        yield qwen
    finally:
        with torch.no_grad():
            embedding.weight[index].copy_(input_old)
            output.weight[index].copy_(output_old)
        qwen.set_input_embeddings(embedding)
        qwen.set_output_embeddings(output)
