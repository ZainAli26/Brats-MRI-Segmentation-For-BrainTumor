"""The ``replica:`` config block — everything the loop needs that the plan doesn't say.

The plan owns the *dataset-derived* decisions (patch, batch, spacing, topology,
batch_dice). This block owns the *training-budget* decisions nnU-Net hard-codes in
``nnUNetTrainer`` (1000 epochs × 250 iterations, SGD 1e-2 / 0.99 / 3e-5, poly 0.9,
33 % forced-foreground, grad-norm clip 12) plus the handful of knobs that exist only
because we are fitting an 11 GB plan onto smaller hardware.

Defaults are nnU-Net's. An experiment that changes one is deliberately *not* a replica
of the native run any more, which is exactly what the exp21+ ablations want — so every
deviation is a named field rather than an edit to the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ReplicaConfig:
    # --- what to replicate -------------------------------------------------------
    plans_json: str = "plans/nnUNetResEncM_11GBPlans.json"
    configuration: str = "3d_fullres"
    preprocessed_dir: str = ".cache/replica/Dataset102_BraTS2024ResEnc"
    store_dtype: str = "float16"   # cache precision; float32 matches nnU-Net exactly

    # --- nnUNetTrainer's fixed budget --------------------------------------------
    num_epochs: int = 1000
    num_iterations_per_epoch: int = 250
    num_val_iterations_per_epoch: int = 50
    initial_lr: float = 1e-2
    weight_decay: float = 3e-5
    momentum: float = 0.99
    nesterov: bool = True
    poly_exponent: float = 0.9
    oversample_foreground_percent: float = 0.33
    grad_clip: float = 12.0
    amp: bool = True
    save_every: int = 50

    # --- loss --------------------------------------------------------------------
    loss: str = "dice_ce"          # dice_ce (nnU-Net) | dice_focal (ablation)
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    focal_gamma: float = 2.0
    deep_supervision: bool = True
    ds_zero_lowest: bool = True    # nnU-Net >= 2.2 zeroes the deepest head
    batch_dice: Optional[bool] = None   # None -> take it from the plan

    # --- fitting the 11 GB plan onto this GPU ------------------------------------
    batch_size: Optional[int] = None      # None -> from the plan (2)
    grad_accum_steps: int = 1             # effective batch stays batch_size
    patch_size: Optional[List[int]] = None  # None -> from the plan
    grad_checkpointing: bool = False

    # --- architecture ablation ----------------------------------------------------
    # None -> build the network the plan names (the replica case). "segresnet_ds" swaps
    # in MONAI's SegResNetDS while keeping every other element of the recipe, so the
    # comparison isolates the architecture. Patch/batch still come from the plan.
    architecture_override: Optional[str] = None
    segresnet: Dict = field(default_factory=dict)

    # --- plumbing ----------------------------------------------------------------
    num_workers: int = 6
    seed: int = 42
    validate_with_full_images: bool = True   # final sliding-window validation pass
    tile_step_size: float = 0.5
    use_mirroring: bool = True

    extra: Dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Dict) -> "ReplicaConfig":
        block = dict(config.get("replica") or {})
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in block.items() if k in known}
        extra = {k: v for k, v in block.items() if k not in known}
        obj = cls(**kwargs, extra=extra)
        obj.validate()
        return obj

    def validate(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.loss not in ("dice_ce", "dice_focal"):
            raise ValueError(f"replica.loss must be dice_ce or dice_focal, got {self.loss!r}")
        if self.architecture_override not in (None, "segresnet_ds"):
            raise ValueError(
                f"replica.architecture_override must be null or 'segresnet_ds', "
                f"got {self.architecture_override!r}"
            )

    def resolve_batch(self, plan_batch_size: int) -> tuple[int, int]:
        """Return ``(effective_batch_size, micro_batch_size)``.

        Gradient accumulation here is *exact*, not an approximation: with
        ``batch_dice=False`` both the Dice term (per-sample mean) and CE (per-voxel mean)
        are means over equally sized micro-batches, and InstanceNorm never mixes samples —
        so N micro-steps of size B/N produce the same gradient as one step of size B. That
        equivalence breaks if ``batch_dice`` is on, which is why the trainer warns.
        """
        batch = int(self.batch_size or plan_batch_size)
        if batch % self.grad_accum_steps != 0:
            raise ValueError(
                f"grad_accum_steps ({self.grad_accum_steps}) must divide the batch size "
                f"({batch}); otherwise micro-batches are unequal and accumulation is no "
                f"longer equivalent to a single large step."
            )
        return batch, batch // self.grad_accum_steps

    def resolved_preprocessed_dir(self) -> Path:
        return Path(self.preprocessed_dir).expanduser()
