"""The custom loop: nnU-Net v2's ``nnUNetTrainer``, rewritten as our own trainer.

Every element of the native recipe is here explicitly, and nothing else is:

  epoch      = exactly 250 optimiser steps sampled with replacement (not a pass over the data)
  optimiser  = SGD(lr 1e-2, momentum 0.99, nesterov, weight_decay 3e-5)
  schedule   = poly, ``lr = 1e-2 * (1 - epoch/1000)**0.9``, stepped per epoch, no warmup
  precision  = autocast + GradScaler, unscale → clip grad-norm to 12 → step
  loss       = Dice(no bg, smooth 1e-5) + CE, deep-supervised with 1/2**i weights
  validation = 50 random patches/epoch → global TP/FP/FN → pseudo-Dice → 0.9 EMA
  checkpoint = best EMA pseudo-Dice, plus ``checkpoint_latest`` every 50 epochs

The one concession to hardware is gradient accumulation: the plan asks for batch 2 at
[128, 192, 128], which does not fit 8 GB, so the loop can run N micro-steps of
``batch/N``. With ``batch_dice=False`` (this plan) that is mathematically the same
gradient, not an approximation — see ``ReplicaConfig.resolve_batch``.

Deliberately *absent*: early stopping, "best full-image val Dice" checkpointing, LR
warmup, cosine restarts. Each of those was in the old loop and each is a divergence from
the run being replicated.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from src.nnunet_replica.augmentation import (
    configure_rotation_and_initial_patch_size,
    get_training_transforms,
    get_validation_transforms,
    mask_channels_for_norm,
)
from src.nnunet_replica.config import ReplicaConfig
from src.nnunet_replica.dataloading import PreprocessedDataset, build_loader
from src.nnunet_replica.logger import ReplicaLogger
from src.nnunet_replica.loss import build_loss
from src.nnunet_replica.lr_scheduler import PolyLRScheduler
from src.nnunet_replica.network import (
    build_network,
    count_parameters,
    set_deep_supervision_enabled,
)
from src.nnunet_replica.plans import Plans


def get_tp_fp_fn(pred_onehot: torch.Tensor, target: torch.Tensor,
                 axes: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard TP/FP/FN per class — the ingredients of nnU-Net's pseudo-Dice."""
    with torch.no_grad():
        if target.ndim == pred_onehot.ndim and target.shape[1] == 1:
            y_onehot = torch.zeros_like(pred_onehot)
            y_onehot.scatter_(1, target.long(), 1)
        else:
            y_onehot = target
    tp = (pred_onehot * y_onehot).sum(dim=list(axes))
    fp = (pred_onehot * (1 - y_onehot)).sum(dim=list(axes))
    fn = ((1 - pred_onehot) * y_onehot).sum(dim=list(axes))
    return tp, fp, fn


class NNUNetReplicaTrainer:
    """nnU-Net v2's training procedure, implemented as a standalone loop."""

    def __init__(
        self,
        config: Dict,
        replica_cfg: ReplicaConfig,
        fold: int,
        train_case_ids: Sequence[str],
        val_case_ids: Sequence[str],
        output_folder: str | Path,
        tracker=None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.cfg = replica_cfg
        self.fold = fold
        self.train_case_ids = list(train_case_ids)
        self.val_case_ids = list(val_case_ids)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.plans = Plans.load(self.cfg.plans_json)
        self.plan_cfg = self.plans.get_configuration(self.cfg.configuration)

        self.patch_size: List[int] = list(self.cfg.patch_size or self.plan_cfg.patch_size)
        self.batch_size, self.micro_batch_size = self.cfg.resolve_batch(self.plan_cfg.batch_size)
        self.batch_dice = (self.plan_cfg.batch_dice if self.cfg.batch_dice is None
                           else bool(self.cfg.batch_dice))

        self.num_input_channels = len(config["data"]["modalities"])
        self.num_classes = int(config["data"]["num_classes"])
        self.foreground_labels = [i for i in range(1, self.num_classes)]

        self.logger = ReplicaLogger(self.output_folder, tracker=tracker)
        self.current_epoch = 0
        self._best_ema: Optional[float] = None

        self.network: Optional[nn.Module] = None
        self.optimizer = self.lr_scheduler = self.loss = None
        self.grad_scaler = None
        self.dataloader_train = self.dataloader_val = None
        self._was_initialized = False

    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        if self._was_initialized:
            return

        # Deep supervision resolutions are derived from the plan's strides — except for
        # an overridden architecture, which brings its own downsampling schedule.
        self.deep_supervision_scales = (
            self.plan_cfg.deep_supervision_scales if self.cfg.deep_supervision else None
        )

        self._ds_zero_lowest = self.cfg.ds_zero_lowest

        if self.cfg.architecture_override == "segresnet_ds":
            from src.nnunet_replica.network import build_segresnet_ds
            sr = dict(self.cfg.segresnet or {})
            plan_heads = len(self.plan_cfg.deep_supervision_scales)
            self.network, scales = build_segresnet_ds(
                self.num_input_channels, self.num_classes,
                init_filters=sr.get("init_filters", 32),
                blocks_down=sr.get("blocks_down", (1, 2, 2, 4, 4)),
                dsdepth=sr.get("dsdepth", plan_heads),
            )
            if self.cfg.deep_supervision:
                self.deep_supervision_scales = scales
                # nnU-Net zeroes its deepest head because it is too coarse to supervise.
                # SegResNetDS simply does not emit that level, so zeroing again would
                # discard a head nnU-Net actually trains on. Dropping the zeroing when the
                # net is exactly one head short makes the supervised weights identical to
                # the plan's non-zero weights ([0.533, 0.267, 0.133, 0.067]) instead of
                # re-normalising over three heads.
                if self.cfg.ds_zero_lowest and len(scales) == plan_heads - 1:
                    self._ds_zero_lowest = False
            self.network = self.network.to(self.device)
        else:
            self.network = build_network(
                self.plan_cfg,
                num_input_channels=self.num_input_channels,
                num_output_classes=self.num_classes,
                deep_supervision=self.cfg.deep_supervision,
                grad_checkpointing=self.cfg.grad_checkpointing,
            ).to(self.device)

        self.optimizer = torch.optim.SGD(
            self.network.parameters(), self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum,
            nesterov=self.cfg.nesterov,
        )
        self.lr_scheduler = PolyLRScheduler(
            self.optimizer, self.cfg.initial_lr, self.cfg.num_epochs, self.cfg.poly_exponent
        )
        self.loss = build_loss(
            batch_dice=self.batch_dice,
            deep_supervision_scales=self.deep_supervision_scales,
            loss_name=self.cfg.loss,
            weight_dice=self.cfg.dice_weight,
            weight_ce=self.cfg.ce_weight,
            focal_gamma=self.cfg.focal_gamma,
            zero_lowest_ds=self._ds_zero_lowest,
        )
        self.grad_scaler = (torch.amp.GradScaler("cuda")
                            if self.device.type == "cuda" else None)

        self._build_dataloaders()
        self._was_initialized = True
        self._print_setup()

    def _build_dataloaders(self) -> None:
        folder = self.cfg.resolved_preprocessed_dir()
        ds_train = PreprocessedDataset(folder, self.train_case_ids)
        ds_val = PreprocessedDataset(folder, self.val_case_ids)

        rotation, do_dummy_2d, initial_patch_size, mirror_axes = \
            configure_rotation_and_initial_patch_size(self.patch_size)
        self.mirror_axes = mirror_axes if self.cfg.use_mirroring else None
        self.initial_patch_size = initial_patch_size

        train_tf = get_training_transforms(
            self.patch_size, rotation, self.deep_supervision_scales, mirror_axes, do_dummy_2d,
            use_mask_for_norm=mask_channels_for_norm(
                self.plan_cfg.use_mask_for_norm, self.num_input_channels
            ),
            order_resampling_data=self.plan_cfg.resampling_order_data,
            order_resampling_seg=self.plan_cfg.resampling_order_seg,
        )
        val_tf = get_validation_transforms(self.deep_supervision_scales)

        self.dataloader_train = build_loader(
            ds_train, self.batch_size, initial_patch_size, self.patch_size,
            self.foreground_labels, train_tf, num_workers=self.cfg.num_workers,
            oversample_foreground_percent=self.cfg.oversample_foreground_percent,
            seed=self.cfg.seed + 100 * self.fold,
        )
        # Validation samples the patch size directly — no rotation inflation.
        self.dataloader_val = build_loader(
            ds_val, self.batch_size, self.patch_size, self.patch_size,
            self.foreground_labels, val_tf,
            num_workers=max(1, self.cfg.num_workers // 2),
            oversample_foreground_percent=self.cfg.oversample_foreground_percent,
            seed=self.cfg.seed + 100 * self.fold + 7,
        )

    def _print_setup(self) -> None:
        p = self.logger.print_to_log_file
        p("#" * 72, add_timestamp=False)
        p(f"nnU-Net replica trainer — fold {self.fold}", add_timestamp=False)
        p(self.plans.summary(self.cfg.configuration), add_timestamp=False)
        p(f"  network params: {count_parameters(self.network):,}", add_timestamp=False)
        p(f"  patch {self.patch_size}  batch {self.batch_size} "
          f"(micro {self.micro_batch_size} x accum {self.cfg.grad_accum_steps})", add_timestamp=False)
        p(f"  sampled patch (rotation-inflated): {self.initial_patch_size}", add_timestamp=False)
        p(f"  epochs {self.cfg.num_epochs} x {self.cfg.num_iterations_per_epoch} iters, "
          f"val {self.cfg.num_val_iterations_per_epoch} iters", add_timestamp=False)
        p(f"  SGD lr {self.cfg.initial_lr} momentum {self.cfg.momentum} "
          f"nesterov {self.cfg.nesterov} wd {self.cfg.weight_decay}, poly^{self.cfg.poly_exponent}",
          add_timestamp=False)
        p(f"  loss {self.cfg.loss}  batch_dice {self.batch_dice}  "
          f"deep supervision {self.cfg.deep_supervision} (zero lowest {self._ds_zero_lowest})",
          add_timestamp=False)
        if self.deep_supervision_scales is not None:
            from src.nnunet_replica.loss import deep_supervision_weights
            w = deep_supervision_weights(len(self.deep_supervision_scales), self._ds_zero_lowest)
            p(f"  DS heads {len(self.deep_supervision_scales)} at "
              f"{[round(s[0], 4) for s in self.deep_supervision_scales]}, "
              f"weights {[round(float(v), 4) for v in w]}", add_timestamp=False)
            if self._ds_zero_lowest != self.cfg.ds_zero_lowest:
                p("  NOTE: ds_zero_lowest overridden to False — this network emits one "
                  "fewer head than the plan, i.e. it already omits the level nnU-Net "
                  "zeroes. Supervised weights now match the plan's non-zero weights.",
                  add_timestamp=False)
        p(f"  train cases {len(self.train_case_ids)}  val cases {len(self.val_case_ids)}",
          add_timestamp=False)
        if self.batch_dice and self.cfg.grad_accum_steps > 1:
            p("  WARNING: batch_dice=True with grad_accum_steps>1 — accumulation is NOT "
              "equivalent to a single large batch, because Dice is pooled over the batch. "
              "Set grad_accum_steps=1 or batch_dice=False for an exact replica.",
              add_timestamp=False)
        p("#" * 72, add_timestamp=False)

    # -------------------------------------------------------------- one step
    def _split_micro(self, data: torch.Tensor, target) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        n = self.cfg.grad_accum_steps
        if n == 1:
            return [(data, target)]
        data_chunks = data.chunk(n, dim=0)
        if isinstance(target, list):
            target_chunks = list(zip(*[t.chunk(n, dim=0) for t in target]))
            return [(d, list(t)) for d, t in zip(data_chunks, target_chunks)]
        return [(d, t) for d, t in zip(data_chunks, target.chunk(n, dim=0))]

    def _compute_loss(self, output, target):
        """Loss over matching (head, target) pairs.

        The planned nets always return one tensor per deep-supervision scale, so this is
        a pass-through for them. An overridden architecture may return a single tensor in
        eval mode (``SegResNetDS`` does); in that case score only the full-resolution head
        with the *unwrapped* loss, so the value stays on the same scale instead of being
        multiplied by the top deep-supervision weight.
        """
        if isinstance(output, (list, tuple)) and isinstance(target, list):
            n = min(len(output), len(target))
            return self.loss(list(output)[:n], target[:n])
        head = output[0] if isinstance(output, (list, tuple)) else output
        tgt = target[0] if isinstance(target, list) else target
        base = getattr(self.loss, "loss", self.loss)   # unwrap DeepSupervisionWrapper
        return base(head, tgt)

    def train_step(self, batch: Dict) -> Dict:
        data = batch["data"]
        target = batch["target"]

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        micro = self._split_micro(data, target)

        for md, mt in micro:
            md = md.to(self.device, non_blocking=True)
            mt = ([t.to(self.device, non_blocking=True) for t in mt]
                  if isinstance(mt, list) else mt.to(self.device, non_blocking=True))
            with torch.autocast(self.device.type, enabled=self.cfg.amp) \
                    if self.device.type == "cuda" else _nullcontext():
                output = self.network(md)
                l = self._compute_loss(output, mt)
                l_scaled = l / len(micro)
            if self.grad_scaler is not None:
                self.grad_scaler.scale(l_scaled).backward()
            else:
                l_scaled.backward()
            total_loss += float(l.detach().cpu())

        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

        return {"loss": total_loss / len(micro)}

    @torch.no_grad()
    def validation_step(self, batch: Dict) -> Dict:
        data, target = batch["data"], batch["target"]
        total_loss, tps, fps, fns = 0.0, [], [], []
        micro = self._split_micro(data, target)

        for md, mt in micro:
            md = md.to(self.device, non_blocking=True)
            mt = ([t.to(self.device, non_blocking=True) for t in mt]
                  if isinstance(mt, list) else mt.to(self.device, non_blocking=True))
            with torch.autocast(self.device.type, enabled=self.cfg.amp) \
                    if self.device.type == "cuda" else _nullcontext():
                output = self.network(md)
                l = self._compute_loss(output, mt)
            total_loss += float(l.detach().cpu())

            # Pseudo-Dice uses only the full-resolution head.
            out = output[0] if isinstance(output, (list, tuple)) else output
            tgt = mt[0] if isinstance(mt, list) else mt

            axes = [0] + list(range(2, out.ndim))
            pred_seg = out.argmax(1)[:, None]
            pred_onehot = torch.zeros(out.shape, device=out.device, dtype=torch.float32)
            pred_onehot.scatter_(1, pred_seg, 1)
            tp, fp, fn = get_tp_fp_fn(pred_onehot, tgt, axes)
            # [1:] drops background — nnU-Net does not score it.
            tps.append(tp[1:].cpu().numpy())
            fps.append(fp[1:].cpu().numpy())
            fns.append(fn[1:].cpu().numpy())

        return {
            "loss": total_loss / len(micro),
            "tp_hard": np.sum(tps, 0), "fp_hard": np.sum(fps, 0), "fn_hard": np.sum(fns, 0),
        }

    # ------------------------------------------------------------ epoch hooks
    def on_train_epoch_start(self) -> None:
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.logger.print_to_log_file("")
        self.logger.print_to_log_file(f"Epoch {self.current_epoch}")
        self.logger.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

    def on_epoch_end(self) -> None:
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        p = self.logger.print_to_log_file
        p("train_loss", np.round(self.logger.logging["train_losses"][self.current_epoch], 4))
        p("val_loss", np.round(self.logger.logging["val_losses"][self.current_epoch], 4))
        p("Pseudo dice", [round(float(i), 4) for i in
                          self.logger.logging["dice_per_class"][self.current_epoch]])
        dt = (self.logger.logging["epoch_end_timestamps"][self.current_epoch]
              - self.logger.logging["epoch_start_timestamps"][self.current_epoch])
        p(f"Epoch time: {np.round(dt, 2)} s")

        if (self.current_epoch + 1) % self.cfg.save_every == 0 \
                and self.current_epoch != self.cfg.num_epochs - 1:
            self.save_checkpoint(self.output_folder / "checkpoint_latest.pth")

        ema = self.logger.logging["ema_fg_dice"][self.current_epoch]
        if self._best_ema is None or ema > self._best_ema:
            self._best_ema = ema
            p(f"Yayy! New best EMA pseudo Dice: {np.round(ema, 4)}")
            self.save_checkpoint(self.output_folder / "checkpoint_best.pth")

        self.logger.plot_progress_png()
        self.logger.save_json()
        self.current_epoch += 1

    # --------------------------------------------------------------- the loop
    def run_training(self) -> float:
        self.initialize()
        train_iter = iter(self.dataloader_train)
        val_iter = iter(self.dataloader_val)

        while self.current_epoch < self.cfg.num_epochs:
            self.logger.log("epoch_start_timestamps", time(), self.current_epoch)
            self.on_train_epoch_start()

            train_losses = []
            for _ in range(self.cfg.num_iterations_per_epoch):
                train_losses.append(self.train_step(next(train_iter))["loss"])
            self.logger.log("train_losses", float(np.mean(train_losses)), self.current_epoch)

            self.network.eval()
            val_losses, tp, fp, fn = [], 0, 0, 0
            for _ in range(self.cfg.num_val_iterations_per_epoch):
                out = self.validation_step(next(val_iter))
                val_losses.append(out["loss"])
                tp = tp + out["tp_hard"]; fp = fp + out["fp_hard"]; fn = fn + out["fn_hard"]

            dice_per_class = [float(2 * i / (2 * i + j + k)) if (2 * i + j + k) > 0 else np.nan
                              for i, j, k in zip(tp, fp, fn)]
            self.logger.log("val_losses", float(np.mean(val_losses)), self.current_epoch)
            self.logger.log("dice_per_class", dice_per_class, self.current_epoch)
            self.logger.log("mean_fg_dice", float(np.nanmean(dice_per_class)), self.current_epoch)

            self.on_epoch_end()

        self.save_checkpoint(self.output_folder / "checkpoint_final.pth")
        self.logger.print_to_log_file(
            f"Training done. Best EMA pseudo Dice: {np.round(self._best_ema or 0.0, 4)}"
        )
        return float(self._best_ema or 0.0)

    # ---------------------------------------------------------- checkpointing
    def save_checkpoint(self, filename: str | Path) -> None:
        torch.save({
            "network_weights": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler else None,
            "logging": self.logger.logging,
            "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1,
            "config": self.config,
            "replica_config": self.cfg.__dict__,
            "patch_size": self.patch_size,
            "num_classes": self.num_classes,
            "num_input_channels": self.num_input_channels,
            "fold": self.fold,
        }, str(filename))

    def load_checkpoint(self, filename: str | Path) -> None:
        self.initialize()
        ckpt = torch.load(str(filename), map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network_weights"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.grad_scaler is not None and ckpt.get("grad_scaler_state") is not None:
            self.grad_scaler.load_state_dict(ckpt["grad_scaler_state"])
        self.logger.logging = ckpt["logging"]
        self._best_ema = ckpt["_best_ema"]
        self.current_epoch = ckpt["current_epoch"]
        self.logger.print_to_log_file(
            f"Resumed from {filename} at epoch {self.current_epoch} "
            f"(best EMA pseudo Dice {np.round(self._best_ema or 0.0, 4)})"
        )

    # ------------------------------------------------- final image validation
    def perform_actual_validation(self, save_segmentations_to: Optional[Path] = None) -> Dict:
        """Full-image sliding-window validation, deep supervision off — nnU-Net's final pass.

        Per-epoch pseudo-Dice is a patch-level proxy used only for checkpoint selection.
        The number that is comparable to the native run comes from here.
        """
        from src.nnunet_replica.inference import predict_case

        set_deep_supervision_enabled(self.network, False)
        self.network.eval()

        folder = self.cfg.resolved_preprocessed_dir()
        ds = PreprocessedDataset(folder, self.val_case_ids)
        regions = self.config["evaluation"]["regions"]
        per_case: List[Dict] = []

        for case_id in self.val_case_ids:
            data, seg, props = ds.load_case(case_id)
            pred = predict_case(
                self.network, np.asarray(data, dtype=np.float32), props,
                self.patch_size, self.num_classes, device=self.device,
                tile_step_size=self.cfg.tile_step_size,
                mirror_axes=self.mirror_axes if self.cfg.use_mirroring else None,
                amp=self.cfg.amp, return_to_original_space=False,
            )
            gt = np.asarray(seg[0])
            gt = np.where(gt < 0, 0, gt)   # -1 marks outside-the-brain

            row = {"case_id": case_id}
            for name, labels in regions.items():
                p_mask = np.isin(pred, labels)
                g_mask = np.isin(gt, labels)
                denom = p_mask.sum() + g_mask.sum()
                row[f"dice_{name}"] = (2.0 * (p_mask & g_mask).sum() / denom) if denom > 0 else np.nan
            per_case.append(row)

            if save_segmentations_to is not None:
                save_segmentations_to = Path(save_segmentations_to)
                save_segmentations_to.mkdir(parents=True, exist_ok=True)
                np.save(save_segmentations_to / f"{case_id}.npy", pred.astype(np.uint8))

        summary = {
            f"dice_{name}": float(np.nanmean([r[f"dice_{name}"] for r in per_case]))
            for name in regions
        }
        summary["mean_region_dice"] = float(np.mean(list(summary.values())))
        summary["num_cases"] = len(per_case)

        with open(self.output_folder / "validation_summary.json", "w") as f:
            json.dump({"summary": summary, "per_case": per_case}, f, indent=2, default=float)
        self.logger.print_to_log_file("Full-image validation:", json.dumps(summary, default=float))

        set_deep_supervision_enabled(self.network, self.cfg.deep_supervision)
        return summary


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
