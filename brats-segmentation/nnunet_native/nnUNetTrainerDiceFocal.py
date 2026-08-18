"""exp22 as native nnU-Net: Dice + Focal(gamma=2) instead of Dice + CE.

exp22 is exp20 with ONE change — the cross-entropy term becomes focal:

    exp20:  loss = softDice(no bg) + CE          (nnUNetTrainer_750epochs, Dataset104)
    exp22:  loss = softDice(no bg) + Focal(g=2)  (this trainer,            Dataset104)

Same dataset, same plans, same splits, same 750-epoch budget; nnU-Net keys the results
directory off the trainer name, so both runs coexist under Dataset104 without collision.
The focal term follows the replica's RobustFocalLoss (src/nnunet_replica/loss.py) with
gamma fixed at 2.0 and no per-class weights; note the replica exp22 YAML trains 1000
epochs — compare this arm only against same-budget (750-epoch) native runs (CLAUDE.md).

WHY focal: on this dataset the loss is dominated by SNFH and background while NETC and
ET are tiny (a 20-case audit found NETC at ~0.004 % of voxels). Focal down-weights easy,
abundant voxels so more of the gradient lands on the rare classes.

INSTALL: nnU-Net resolves trainer names from its own package tree, so copy this file in
before `-tr nnUNetTrainerDiceFocal_750epochs` resolves:

    cp nnunet_native/nnUNetTrainerDiceFocal.py \
       "$(python3 -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')"/training/nnUNetTrainer/variants/loss/

A pod rebuild or nnunetv2 reinstall silently removes the copy — this repo file is the
source of truth, so re-copy (the training driver should do it idempotently before every
exp22 run). Alternatively, nnunetv2 >= 2.8 can load trainers from an external directory
via the `nnUNet_extTrainer` env var; point it at a directory containing ONLY trainer
modules if you use it.
"""
import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_750epochs,
)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017), drop-in for RobustCrossEntropyLoss.

    Computed from the fused cross-entropy kernel: ce = -log pt, so
    focal = (1 - pt)^gamma * ce with pt = exp(-ce). This keeps the memory profile close
    to the CE it replaces (the hand-rolled log_softmax+gather form retains several
    full-resolution intermediates per deep-supervision scale for backward).

    (1 - pt) is clamped away from 0: for non-even gamma, d/dx x^gamma is unbounded at 0,
    so a single saturated voxel (pt == 1.0 exactly) would emit inf/nan gradients. The
    clamp only touches voxels whose ce is already ~0.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # DC_and_CE_loss always calls self.ce(net_output, target[:, 0]); this squeeze
        # exists only for interface parity with RobustCrossEntropyLoss.
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        ce = F.cross_entropy(input, target.long(), reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt).clamp_min(1e-6) ** self.gamma) * ce).mean()


class nnUNetTrainerDiceFocal_750epochs(nnUNetTrainer_750epochs):
    """Identical to nnUNetTrainer_750epochs except the CE term is Focal(gamma=2)."""

    def _build_loss(self):
        # exp22 trains on labels, not regions; a region-based dataset would take the
        # DC_and_BCE branch upstream and never use focal, so refuse it loudly. Same for
        # ignore_label: DC_and_CE_loss wires it into the CE term as ignore_index, which
        # the FocalLoss swap below would silently drop.
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerDiceFocal_750epochs only supports label-based training"
        assert self.label_manager.ignore_label is None, \
            "nnUNetTrainerDiceFocal_750epochs does not support ignore_label"

        # Reuse the parent recipe verbatim (dice kwargs, torch.compile placement, deep
        # supervision weight ladder) and swap only the CE module, so upstream changes to
        # the recipe keep applying to exp20 and exp22 alike. DeepSupervisionWrapper
        # stores the compound loss as .loss, and _do_i_compile only compiles .dc.
        loss = super()._build_loss()
        compound = loss.loss if self.enable_deep_supervision else loss
        compound.ce = FocalLoss(gamma=2.0)
        return loss
