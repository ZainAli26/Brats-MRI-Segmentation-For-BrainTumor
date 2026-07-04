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
    # nnU-Net v2 computes Dice over the whole batch (pooled), which stabilizes the
    # gradient for small/often-absent classes (ET, NETC). Off by default.
    batch_dice = config["training"].get("batch_dice", False)

    if loss_name == "dice_ce":
        loss_fn = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            batch=batch_dice,
            lambda_dice=config["training"]["dice_weight"],
            lambda_ce=config["training"]["ce_weight"],
        )
    elif loss_name == "dice_focal":
        loss_fn = DiceFocalLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            batch=batch_dice,
            lambda_dice=config["training"]["dice_weight"],
            lambda_focal=config["training"]["ce_weight"],
            gamma=2.0,
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    return loss_fn


class DeepSupervisionLoss(nn.Module):
    """Wrapper for deep supervision loss (used with nnU-Net v2 / DynUNet).

    nnU-Net-exact mode (the default):
      * weights = [1, 1/2, 1/4, ...]; the LOWEST-resolution head is zeroed
        (nnU-Net sets weights[-1] = 0) and the rest are renormalized to sum 1.
      * the loss at each head is computed at the head's OWN (lower) resolution by
        DOWNSAMPLING the ground-truth label with nearest-neighbour interpolation
        (nnU-Net's DownsampleSegForDSTransform), NOT by upsampling the prediction.

    Set downsample_target=False to fall back to the legacy behaviour (upsample the
    prediction to full resolution), and zero_lowest=False to keep every head active.
    """

    def __init__(self, base_loss: nn.Module, weights: list = None,
                 downsample_target: bool = True, zero_lowest: bool = True):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights
        self.downsample_target = downsample_target
        self.zero_lowest = zero_lowest

    def _resolve_weights(self, n: int):
        if self.weights is not None:
            weights = list(self.weights[:n])
        else:
            weights = [1.0 / (2 ** i) for i in range(n)]
        # nnU-Net zeroes the lowest-resolution (deepest) head before normalizing.
        if self.zero_lowest and n > 1:
            weights[-1] = 0.0
        w_sum = sum(weights) or 1.0
        return [w / w_sum for w in weights]

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

        from torch.nn.functional import interpolate

        n = len(predictions)
        weights = self._resolve_weights(n)

        total_loss = 0
        for i, pred in enumerate(predictions):
            if weights[i] == 0.0:
                continue  # skip the zeroed (deepest) head entirely
            if pred.shape[2:] != target.shape[2:]:
                if self.downsample_target:
                    # nnU-Net-exact: shrink the label to the head's resolution
                    # (nearest preserves discrete class ids; the base loss one-hots).
                    tgt = interpolate(target.float(), size=pred.shape[2:], mode="nearest")
                    total_loss += weights[i] * self.base_loss(pred, tgt)
                    continue
                # Legacy: upsample the prediction to full resolution instead.
                pred = interpolate(pred, size=target.shape[2:], mode="trilinear", align_corners=False)
            total_loss += weights[i] * self.base_loss(pred, target)

        return total_loss
