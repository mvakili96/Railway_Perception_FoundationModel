"""Shared, unchanged LISA mask losses; no legacy language-model imports."""

import torch
from torch.nn import functional as F

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

