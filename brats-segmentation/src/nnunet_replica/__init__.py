"""A from-scratch reimplementation of nnU-Net v2's training procedure.

The goal is narrow and specific: run the *same recipe* the native `nnUNetv2_train`
would run for a given `plans.json` — here ``nnUNetResEncM_11GBPlans`` / ``3d_fullres``
— inside a training loop we own, so the loop can then be modified for experiments
while the baseline provably reproduces the native numbers.

    plans.py         parse plans.json (patch/batch/topology/batch_dice)
    preprocessing.py crop-to-nonzero → masked z-score → resample → class locations
    dataloading.py   nnUNetDataLoader3D patch sampling (33 % forced foreground)
    augmentation.py  nnU-Net's batchgenerators pipeline, transform for transform
    network.py       build from the plan + InitWeights_He(1e-2)
    loss.py          Dice(no bg) + CE under deep supervision
    lr_scheduler.py  poly LR
    trainer.py       250-iteration epochs, SGD 0.99 nesterov, EMA-pseudo-Dice checkpointing
    inference.py     Gaussian sliding window + mirroring TTA

Nothing here imports ``nnunetv2`` at training time; the only third-party pieces are
``batchgenerators`` (augmentation) and ``dynamic_network_architectures`` (the U-Net
classes the plan names), both of which the native run also uses.
"""

from src.nnunet_replica.config import ReplicaConfig
from src.nnunet_replica.plans import Plans

__all__ = ["ReplicaConfig", "Plans"]
