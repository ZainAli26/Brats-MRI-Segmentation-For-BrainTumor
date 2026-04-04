"""Loss functions for BraTS segmentation training."""

import torch
import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss


def create_loss(config: dict) -> nn.Module:
    """Create loss function based on config.

    Args:
        config: Full config dict.

    Returns:
        Loss module.
    """
    loss_name = config["training"]["loss"]
    num_classes = config["data"]["num_classes"]

    if loss_name == "dice_ce":
        loss_fn = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=config["training"]["dice_weight"],
            lambda_ce=config["training"]["ce_weight"],
        )
    elif loss_name == "dice_focal":
        loss_fn = DiceFocalLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=config["training"]["dice_weight"],
            lambda_focal=config["training"]["ce_weight"],
            gamma=2.0,
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    return loss_fn


class DeepSupervisionLoss(nn.Module):
    """Wrapper for deep supervision loss (used with DynUNet)."""

    def __init__(self, base_loss: nn.Module, weights: list = None):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, predictions, target):
        # Handle different deep supervision output formats:
        #   - MONAI DynUNet: stacked tensor [B, heads, C, H, W, D]
        #   - nnU-Net v2:    list/tuple of tensors [B, C, H, W, D]
        #   - No deep sup:   single tensor [B, C, H, W, D]

        if isinstance(predictions, torch.Tensor) and predictions.ndim == target.ndim + 1:
            # DynUNet stacked format: [B, heads, C, H, W, D] -> list of [B, C, H, W, D]
            predictions = [predictions[:, i] for i in range(predictions.shape[1])]
        elif isinstance(predictions, torch.Tensor) and predictions.ndim == target.ndim:
            # No deep supervision — single tensor
            return self.base_loss(predictions, target)

        if not isinstance(predictions, (list, tuple)):
            return self.base_loss(predictions, target)

        n = len(predictions)
        weights = self.weights or [1.0 / (2 ** i) for i in range(n)]
        # Normalize weights
        w_sum = sum(weights[:n])
        weights = [w / w_sum for w in weights[:n]]

        total_loss = 0
        for i, pred in enumerate(predictions):
            if pred.shape[2:] != target.shape[2:]:
                # Deep supervision outputs may be at lower resolution
                from torch.nn.functional import interpolate
                pred = interpolate(pred, size=target.shape[2:], mode="trilinear", align_corners=False)
            total_loss += weights[i] * self.base_loss(pred, target)

        return total_loss
