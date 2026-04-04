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
        if model_name == "nnunet_v2":
            uses_deep_sup = config["model"]["nnunet_v2"].get("deep_supervision", False)
        elif model_name == "dynunet":
            uses_deep_sup = config["model"].get("dynunet", {}).get("deep_supervision", False)

        if uses_deep_sup:
            self.loss_fn = DeepSupervisionLoss(loss_fn)
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
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=train_cfg["epochs"], eta_min=1e-7
            )

        # Mixed precision
        self.use_amp = train_cfg["amp"] and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_accum_steps = train_cfg.get("grad_accum_steps", 1)

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
                mean_dice = val_metrics["mean_dice"]

                self.tracker.log_scalar("val/mean_dice", mean_dice, epoch)
                for region, dice in val_metrics["region_dice"].items():
                    self.tracker.log_scalar(f"val/dice_{region}", dice, epoch)

                # Checkpoint
                if mean_dice > self.best_val_dice:
                    self.best_val_dice = mean_dice
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, mean_dice, is_best=True)
                    console.print(f"  [green]New best: {mean_dice:.4f}[/green]")
                else:
                    self.patience_counter += 1
                    if not self.config["experiment"]["save_best_only"]:
                        self._save_checkpoint(epoch, mean_dice, is_best=False)

                # Early stopping
                if self.patience_counter >= self.patience:
                    console.print(f"\n[yellow]Early stopping at epoch {epoch} (patience={self.patience})[/yellow]")
                    break

        console.print(f"\n[bold green]Training complete. Best val Dice: {self.best_val_dice:.4f}[/bold green]")
        return self.best_val_dice

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        step = 0

        with Progress(
            SpinnerColumn(),
            TextColumn(f"Epoch {epoch}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("training", total=len(self.train_loader))

            for batch_idx, batch_data in enumerate(self.train_loader):
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item() * self.grad_accum_steps
                step += 1
                progress.update(task, advance=1)

        avg_loss = epoch_loss / max(step, 1)

        if epoch % self.config["experiment"].get("log_interval", 10) == 0:
            console.print(f"  Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        return avg_loss

    def _validate(self, epoch: int) -> dict:
        """Run validation with sliding window inference."""
        self.model.eval()
        self.dice_metric.reset()

        regions = self.config["evaluation"]["regions"]

        # Per-region Dice accumulators
        region_dice_sums = {r: 0.0 for r in regions}
        n_samples = 0

        with torch.no_grad():
            for batch_data in self.val_loader:
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

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

                self.dice_metric(y_pred=outputs_onehot, y=labels_onehot)

                # Compute region Dice
                for pred_oh, lab_oh in zip(outputs_onehot, labels_onehot):
                    for region_name, label_indices in regions.items():
                        pred_region = torch.zeros_like(pred_oh[0])
                        lab_region = torch.zeros_like(lab_oh[0])
                        for idx in label_indices:
                            pred_region = torch.logical_or(pred_region, pred_oh[idx])
                            lab_region = torch.logical_or(lab_region, lab_oh[idx])
                        # Dice
                        intersection = (pred_region & lab_region).sum().float()
                        union = pred_region.sum().float() + lab_region.sum().float()
                        dice = (2.0 * intersection / (union + 1e-7)).item()
                        region_dice_sums[region_name] += dice
                    n_samples += 1

        # Per-class dice from MONAI metric
        class_dice = self.dice_metric.aggregate()
        mean_dice = class_dice.mean().item()

        region_dice = {r: region_dice_sums[r] / max(n_samples, 1) for r in regions}

        console.print(
            f"  [cyan]Val Epoch {epoch}[/cyan] | "
            f"Mean Dice: {mean_dice:.4f} | "
            + " | ".join(f"{r}: {d:.4f}" for r, d in region_dice.items())
        )

        return {"mean_dice": mean_dice, "region_dice": region_dice}

    def _save_checkpoint(self, epoch: int, val_dice: float, is_best: bool):
        """Save model checkpoint."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_dice": val_dice,
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
