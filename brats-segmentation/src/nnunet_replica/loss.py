"""nnU-Net v2's loss, ported: soft Dice + cross-entropy under deep supervision.

Faithful port of ``MemoryEfficientSoftDiceLoss``, ``RobustCrossEntropyLoss``,
``DC_and_CE_loss`` and ``DeepSupervisionWrapper``. The details that matter:

* Dice **excludes background** (``do_bg=False``) while CE covers every class.
* ``smooth = 1e-5`` sits in both numerator and denominator, and the denominator is
  clamped at 1e-8 — an absent class therefore scores Dice ≈ 1, not 0, so it does not
  drag the loss down the way MONAI's default does.
* ``batch_dice`` comes from the plan (**false** for this 3d_fullres config): Dice is
  computed per sample and averaged, not pooled over the batch.
* Deep-supervision weights are ``1/2**i`` with the lowest-resolution head zeroed, then
  renormalised to sum to 1.

The zeroing is version-dependent: nnU-Net ≤2.1 kept every head, ≥2.2 sets
``weights[-1] = 0``. The plan being replicated was produced by a ResEnc-M planner
(≥2.4), so zeroing is the default here; ``zero_lowest=False`` restores the old behaviour.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """CE that accepts targets shaped (B, 1, X, Y, Z) as well as (B, X, Y, Z)."""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class MemoryEfficientSoftDiceLoss(nn.Module):
    """Port of nnU-Net's soft Dice (builds the one-hot target under ``no_grad``)."""

    def __init__(self, apply_nonlin: Optional[Callable] = None, batch_dice: bool = False,
                 do_bg: bool = True, smooth: float = 1.0):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if not self.do_bg:
            x = x[:, 1:]

        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))
            if all(i == j for i, j in zip(shp_x, shp_y)):
                y_onehot = y                      # already one-hot
            else:
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
        sum_pred = x.sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / torch.clip(sum_gt + sum_pred + self.smooth, 1e-8)
        return -dc.mean()


class RobustFocalLoss(nn.Module):
    """Multi-class focal loss, drop-in for CE in the loss-ablation experiments.

    Not part of nnU-Net's default recipe — it is here so the exp24/25-style
    "Dice+CE → Dice+Focal" ablation can run inside the replica without changing
    anything else about the loop.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()
        logp = torch.log_softmax(input, dim=1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight.to(loss.device)[target]
        return loss.mean()


class DC_and_CE_loss(nn.Module):
    """``weight_dice * softDice + weight_ce * CE``, as in nnU-Net."""

    def __init__(self, soft_dice_kwargs: dict, ce_kwargs: dict,
                 weight_ce: float = 1.0, weight_dice: float = 1.0,
                 dice_class=MemoryEfficientSoftDiceLoss, ce_class=RobustCrossEntropyLoss):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = ce_class(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0.0
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0.0
        return self.weight_ce * ce_loss + self.weight_dice * dc_loss


class DeepSupervisionWrapper(nn.Module):
    """Apply a loss to every decoder head and combine with fixed weights."""

    def __init__(self, loss: nn.Module, weight_factors: Optional[Sequence[float]] = None):
        super().__init__()
        self.loss = loss
        self.weight_factors = weight_factors

    def forward(self, *args):
        for a in args:
            assert isinstance(a, (tuple, list)), f"all args must be list/tuple, got {type(a)}"
        weights = self.weight_factors if self.weight_factors is not None else [1] * len(args[0])
        total = weights[0] * self.loss(*[a[0] for a in args])
        for i, inputs in enumerate(zip(*args)):
            if i == 0 or weights[i] == 0:
                continue
            total = total + weights[i] * self.loss(*inputs)
        return total


def deep_supervision_weights(num_heads: int, zero_lowest: bool = True) -> np.ndarray:
    """``[1, 1/2, 1/4, ...]``, lowest-resolution head optionally zeroed, sum-normalised."""
    weights = np.array([1 / (2 ** i) for i in range(num_heads)], dtype=np.float64)
    if zero_lowest and num_heads > 1:
        weights[-1] = 0
    return weights / weights.sum()


def build_loss(
    batch_dice: bool,
    deep_supervision_scales: Optional[Sequence],
    loss_name: str = "dice_ce",
    weight_dice: float = 1.0,
    weight_ce: float = 1.0,
    focal_gamma: float = 2.0,
    zero_lowest_ds: bool = True,
    smooth: float = 1e-5,
) -> nn.Module:
    """Build the (optionally deep-supervised) loss the trainer optimises."""
    dice_kwargs = {"batch_dice": batch_dice, "smooth": smooth, "do_bg": False}
    if loss_name == "dice_ce":
        loss: nn.Module = DC_and_CE_loss(dice_kwargs, {}, weight_ce=weight_ce,
                                         weight_dice=weight_dice)
    elif loss_name == "dice_focal":
        loss = DC_and_CE_loss(dice_kwargs, {"gamma": focal_gamma}, weight_ce=weight_ce,
                              weight_dice=weight_dice, ce_class=RobustFocalLoss)
    else:
        raise ValueError(f"Unknown loss '{loss_name}' (expected dice_ce or dice_focal)")

    if deep_supervision_scales is not None:
        weights = deep_supervision_weights(len(deep_supervision_scales), zero_lowest_ds)
        loss = DeepSupervisionWrapper(loss, weights)
    return loss
