"""Training loop for BraTS segmentation models."""

import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from src.training.losses import DeepSupervisionLoss
from src.utils import inference_wrapper
from src.utils.experiment import ExperimentTracker

console = Console()


class Trainer:
    """Handles training, validation, checkpointing, and logging."""

    def __init__(self, model, loss_fn, config, dataloaders, tracker: ExperimentTracker):
        self.model = model
        self.config = config
        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        self.tracker = tracker
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Wrap loss for deep supervision (nnU-Net v2 or DynUNet)
        model_name = config["model"]["name"]
        uses_deep_sup = False
        if model_name.startswith("nnunet_v2"):
            uses_deep_sup = config["model"]["nnunet_v2"].get("deep_supervision", False)
        elif model_name == "dynunet":
            uses_deep_sup = config["model"].get("dynunet", {}).get("deep_supervision", False)

        if uses_deep_sup:
            # nnU-Net-exact deep supervision by default: downsample the GT label to
            # each head's resolution and zero the lowest-resolution head's weight.
            # Both are overridable from the training config for ablations.
            self.loss_fn = DeepSupervisionLoss(
                loss_fn,
                downsample_target=config["training"].get("ds_downsample_target", True),
                zero_lowest=config["training"].get("ds_zero_lowest", True),
            )
        else:
            self.loss_fn = loss_fn

        self.model = self.model.to(self.device)

        # Optimizer
        train_cfg = config["training"]
        if train_cfg["optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=train_cfg["learning_rate"],
                weight_decay=train_cfg["weight_decay"],
            )
        elif train_cfg["optimizer"] == "sgd":
            # nnU-Net v2 recipe: SGD with high Nesterov momentum.
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=train_cfg["learning_rate"],
                momentum=train_cfg.get("momentum", 0.99),
                nesterov=train_cfg.get("nesterov", True),
                weight_decay=train_cfg["weight_decay"],
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=train_cfg["learning_rate"],
                weight_decay=train_cfg["weight_decay"],
            )

        # Scheduler
        if train_cfg["scheduler"] == "cosine_warm_restarts":
            sp = train_cfg["scheduler_params"]
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=sp["T_0"], T_mult=sp["T_mult"], eta_min=sp["eta_min"]
            )
        elif train_cfg["scheduler"] == "poly":
            # nnU-Net v2 recipe: polynomial decay  lr = lr0 * (1 - e/E)^exponent,
            # with an optional linear LR warmup (warmup_epochs) — warmup ramps lr from
            # warmup_start_factor*lr0 up to lr0, which prevents the from-scratch
            # background-collapse SGD@1e-2 hits at batch size 1.
            exponent = train_cfg.get("poly_exponent", 0.9)
            total_epochs = max(train_cfg["epochs"], 1)
            warmup = int(train_cfg.get("warmup_epochs", 0))
            warmup_start = float(train_cfg.get("warmup_start_factor", 0.01))

            def _poly_with_warmup(ep, _w=warmup, _ws=warmup_start,
                                  _E=total_epochs, _x=exponent):
                if _w > 0 and ep < _w:
                    return _ws + (1.0 - _ws) * (ep + 1) / _w
                prog = (ep - _w) / max(1, _E - _w)
                return (1.0 - min(prog, 1.0)) ** _x

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_poly_with_warmup
            )
        elif train_cfg["scheduler"] == "constant":
            # Fixed LR for the whole run (used by the overfit sanity check — a decaying
            # LR stalls memorization before it fully fits).
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda _e: 1.0
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=train_cfg["epochs"], eta_min=1e-7
            )

        # Mixed precision
        self.use_amp = train_cfg["amp"] and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_accum_steps = train_cfg.get("grad_accum_steps", 1)

        # nnU-Net-style fixed-iteration epochs. When set, an "epoch" is exactly
        # `iters_per_epoch` OPTIMIZER STEPS (nnU-Net uses 250), sampling patches with
        # replacement, instead of one full pass over the dataset. This decouples the
        # training budget from dataset size so it matches nnU-Net's regime. None ->
        # legacy full-pass behaviour. `_batch_gen` is the persistent cycling iterator.
        self.iters_per_epoch = train_cfg.get("iters_per_epoch")
        self._batch_gen = None

        # Gradient-norm clipping (nnU-Net v2 default = 12). Bounds exploding gradients
        # so a hot LR + AMP can't blow weights up to NaN and collapse to all-background.
        # Set max_grad_norm: 0 (or null) in the config to disable.
        self.max_grad_norm = train_cfg.get("max_grad_norm", 12.0)

        # Metrics for validation
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        self.post_pred = AsDiscrete(argmax=True, to_onehot=config["data"]["num_classes"])
        self.post_label = AsDiscrete(to_onehot=config["data"]["num_classes"])

        # Sliding window params
        self.sw_batch_size = train_cfg["sw_batch_size"]
        self.sw_overlap = train_cfg["sw_overlap"]
        self.spatial_size = config["preprocessing"]["spatial_size"]

        # Early stopping
        self.best_val_dice = 0.0
        self.patience_counter = 0
        self.patience = train_cfg["early_stopping_patience"]
        self.start_epoch = 1

    def train(self):
        """Run full training loop."""
        train_cfg = self.config["training"]
        epochs = train_cfg["epochs"]
        val_interval = train_cfg["val_interval"]

        console.print(f"\n[bold cyan]Starting training for {epochs} epochs on {self.device}[/bold cyan]")
        if self.start_epoch > 1:
            console.print(f"[bold yellow]Resuming from epoch {self.start_epoch} (best dice: {self.best_val_dice:.4f})[/bold yellow]")
        console.print(f"[dim]AMP: {self.use_amp} | Val every {val_interval} epochs | Patience: {self.patience}[/dim]\n")

        for epoch in range(self.start_epoch, epochs + 1):
            # Train one epoch
            train_loss = self._train_epoch(epoch)
            self.scheduler.step()

            self.tracker.log_scalar("train/loss", train_loss, epoch)
            self.tracker.log_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)

            # Validate
            if epoch % val_interval == 0:
                val_metrics = self._validate(epoch)
                region_dice = val_metrics["region_dice"]

                # Log all metrics
                self.tracker.log_scalar("val/mean_class_dice", val_metrics["mean_dice"], epoch)
                for region, dice in region_dice.items():
                    self.tracker.log_scalar(f"val/dice_{region}", dice, epoch)

                # Use mean region Dice (ET, TC, WT) as the primary metric
                # This matches BraTS challenge reporting and paper conventions
                mean_region_dice = sum(region_dice.values()) / len(region_dice)
                self.tracker.log_scalar("val/mean_region_dice", mean_region_dice, epoch)

                # Checkpoint based on mean region Dice
                if mean_region_dice > self.best_val_dice:
                    self.best_val_dice = mean_region_dice
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, mean_region_dice, is_best=True,
                                          region_dice=region_dice)
                    console.print(
                        f"  [green]New best mean region Dice: {mean_region_dice:.4f} "
                        f"(ET={region_dice.get('ET',0):.4f} TC={region_dice.get('TC',0):.4f} "
                        f"WT={region_dice.get('WT',0):.4f})[/green]"
                    )
                else:
                    self.patience_counter += 1
                    if not self.config["experiment"]["save_best_only"]:
                        self._save_checkpoint(epoch, mean_region_dice, is_best=False,
                                              region_dice=region_dice)

                # Early stopping
                if self.patience_counter >= self.patience:
                    console.print(f"\n[yellow]Early stopping at epoch {epoch} (patience={self.patience})[/yellow]")
                    break

        console.print(f"\n[bold green]Training complete. Best val Dice: {self.best_val_dice:.4f}[/bold green]")
        return self.best_val_dice

    def _iter_train_batches(self):
        """Yield training batches indefinitely, recreating the loader iterator
        each time it is exhausted (sampling-with-replacement across epochs).
        Used only by the fixed-iteration epoch mode."""
        while True:
            for batch in self.train_loader:
                yield batch

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = torch.zeros((), device=self.device)
        step = 0

        # Fixed-iteration mode (nnU-Net): run exactly iters_per_epoch optimizer steps,
        # i.e. iters_per_epoch * grad_accum_steps forward passes, cycling the loader.
        if self.iters_per_epoch:
            num_forward = self.iters_per_epoch * self.grad_accum_steps
            if self._batch_gen is None:
                self._batch_gen = self._iter_train_batches()
            batch_iter = ((i, next(self._batch_gen)) for i in range(num_forward))
            total = num_forward
        else:
            batch_iter = enumerate(self.train_loader)
            total = len(self.train_loader)

        with Progress(
            SpinnerColumn(),
            TextColumn(f"Epoch {epoch}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("training", total=total)

            for batch_idx, batch_data in batch_iter:
                images = batch_data["image"].to(self.device, non_blocking=True)
                labels = batch_data["label"].to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.max_grad_norm:
                        # nnU-Net clips gradient norm to 12. Must unscale before
                        # clipping under AMP; bounds exploding grads -> no NaN blow-up.
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.detach() * self.grad_accum_steps
                step += 1
                progress.update(task, advance=1)

        avg_loss = (epoch_loss / max(step, 1)).item()

        if epoch % self.config["experiment"].get("log_interval", 10) == 0:
            console.print(f"  Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        return avg_loss

    def _validate(self, epoch: int) -> dict:
        """Run validation with sliding window inference."""
        self.model.eval()
        self.dice_metric.reset()

        regions = self.config["evaluation"]["regions"]

        # Per-region Dice accumulators (kept on GPU to avoid per-sample sync)
        region_dice_sums = {r: torch.zeros((), device=self.device) for r in regions}
        n_samples = 0
        # Diagnostic: total predicted voxels per class across the val set. If everything
        # lands in class 0, the model has collapsed to all-background (Dice -> 0.00).
        pred_voxels = None

        with torch.no_grad():
            for batch_data in self.val_loader:
                images = batch_data["image"].to(self.device, non_blocking=True)
                labels = batch_data["label"].to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    outputs = sliding_window_inference(
                        images, self.spatial_size, self.sw_batch_size,
                        inference_wrapper(self.model), overlap=self.sw_overlap
                    )

                # Post-process
                outputs_list = decollate_batch(outputs)
                labels_list = decollate_batch(labels)

                outputs_onehot = [self.post_pred(o) for o in outputs_list]
                labels_onehot = [self.post_label(l) for l in labels_list]

                # Accumulate predicted voxels/class (collapse detector)
                for pred_oh in outputs_onehot:
                    counts = pred_oh.sum(dim=(1, 2, 3))
                    pred_voxels = counts if pred_voxels is None else pred_voxels + counts

                self.dice_metric(y_pred=outputs_onehot, y=labels_onehot)

                # Compute region Dice (vectorized — no per-sample .item() sync)
                for pred_oh, lab_oh in zip(outputs_onehot, labels_onehot):
                    for region_name, label_indices in regions.items():
                        pred_region = pred_oh[label_indices].any(dim=0)
                        lab_region = lab_oh[label_indices].any(dim=0)
                        intersection = (pred_region & lab_region).sum().float()
                        union = pred_region.sum().float() + lab_region.sum().float()
                        region_dice_sums[region_name] += 2.0 * intersection / (union + 1e-7)
                    n_samples += 1

        # Per-class dice from MONAI metric
        class_dice = self.dice_metric.aggregate()
        mean_dice = class_dice.mean().item()

        region_dice = {r: (region_dice_sums[r] / max(n_samples, 1)).item() for r in regions}

        mean_region_dice = sum(region_dice.values()) / len(region_dice)
        console.print(
            f"  [cyan]Val Epoch {epoch}[/cyan] | "
            f"Region Dice: {mean_region_dice:.4f} | "
            + " | ".join(f"{r}: {d:.4f}" for r, d in region_dice.items())
            + f" | Class Dice: {mean_dice:.4f}"
        )
        if pred_voxels is not None:
            counts = [int(v) for v in pred_voxels.tolist()]
            collapsed = sum(counts[1:]) == 0  # no foreground predicted at all
            flag = " [red]<- COLLAPSE: all background[/red]" if collapsed else ""
            console.print(f"  [dim]Pred voxels/class {counts}[/dim]{flag}")

        return {"mean_dice": mean_dice, "region_dice": region_dice}

    def _save_checkpoint(self, epoch: int, val_dice: float, is_best: bool,
                         region_dice: dict = None):
        """Save model checkpoint."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_dice": val_dice,
            "region_dice": region_dice or {},
            "config": self.config,
        }
        if is_best:
            path = self.tracker.run_dir / "best_model.pth"
        else:
            path = self.tracker.run_dir / f"checkpoint_epoch{epoch}.pth"
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        """Load a saved checkpoint and restore full training state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

        # Restore optimizer, scheduler, and training state for proper resume
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "val_dice" in ckpt:
            self.best_val_dice = ckpt["val_dice"]
        if "epoch" in ckpt:
            self.start_epoch = ckpt["epoch"] + 1

        console.print(f"[green]Resumed from {path} — epoch {ckpt['epoch']}, best dice {ckpt.get('val_dice', 0):.4f}[/green]")
        return ckpt
