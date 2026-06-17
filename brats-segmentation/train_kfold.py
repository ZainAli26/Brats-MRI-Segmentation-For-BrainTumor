#!/usr/bin/env python3
"""BraTS Segmentation - 5-Fold Cross-Validation Training.

Trains on ALL data using K-fold CV at patient level. Each fold trains on
~80% of patients and validates on ~20%. Final metrics are averaged across folds.

Usage:
    python train_kfold.py --config experiments/exp17_nnunet_v2_5fold.yaml
    python train_kfold.py --config experiments/exp17_nnunet_v2_5fold.yaml --fold 0  # single fold
    python train_kfold.py --config experiments/exp17_nnunet_v2_5fold.yaml --data_dir /data/Brats2024/training_data1_v2
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import pandas as pd
from rich.console import Console

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
from rich.panel import Panel
from rich.table import Table

from src.utils.experiment import load_config, ExperimentTracker
from src.data.splits import create_kfold_splits
from src.data.dataset import get_dataloaders
from src.data.preprocessing import get_train_transforms, get_val_transforms
from src.models.factory import create_model
from src.training.losses import create_loss
from src.training.trainer import Trainer

console = Console()


def train_fold(config, fold_idx, train_cases, val_cases, args):
    """Train a single fold."""
    console.print(Panel.fit(
        f"[bold cyan]Fold {fold_idx} / {config['data']['n_folds']}[/bold cyan]\n"
        f"[dim]Model: {config['model']['name']} | "
        f"Train: {len(train_cases)} cases | Val: {len(val_cases)} cases[/dim]",
        border_style="bright_blue"
    ))

    # Create fold-specific tracker
    # If resuming, reuse the existing fold directory
    resume_ckpt = None
    if args.resume_dir:
        resume_ckpt = Path(args.resume_dir) / "best_model.pth"
        if resume_ckpt.exists():
            tracker = ExperimentTracker(config, config_path=args.config,
                                        resume_dir=str(args.resume_dir))
        else:
            resume_ckpt = None

    if resume_ckpt is None:
        tracker = ExperimentTracker(config, config_path=args.config)
        # Rename run dir to include fold number
        fold_run_dir = tracker.run_dir.parent / f"{tracker.run_dir.name}_fold{fold_idx}"
        tracker.run_dir.rename(fold_run_dir)
        tracker.run_dir = fold_run_dir
        tracker.checkpoints_dir = fold_run_dir / "checkpoints"
        tracker.viz_dir = fold_run_dir / "visualizations"
        tracker.logs_dir = fold_run_dir / "logs"
        for d in [tracker.checkpoints_dir, tracker.viz_dir, tracker.logs_dir]:
            d.mkdir(exist_ok=True)
        # Reinit tensorboard writer for new path
        from torch.utils.tensorboard import SummaryWriter
        tracker.writer = SummaryWriter(log_dir=str(tracker.logs_dir))
        tracker.log_file = fold_run_dir / "training.log"

    # Build transforms
    modalities = config["data"]["modalities"]
    label_map = {int(k): int(v) for k, v in config["data"]["label_map"].items()}
    spatial_size = config["preprocessing"]["spatial_size"]
    aug_config = config["preprocessing"]["augmentation"]

    train_transform = get_train_transforms(spatial_size, modalities, label_map, aug_config)
    val_transform = get_val_transforms(spatial_size, modalities, label_map)

    # Create dataloaders (no test set in k-fold — all data used for train/val)
    dataloaders = get_dataloaders(
        train_cases, val_cases, [],  # empty test set
        modalities=modalities,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    console.print(f"  Train: {len(dataloaders['train'].dataset)} | Val: {len(dataloaders['val'].dataset)}")

    # Create model (fresh weights each fold)
    model = create_model(config)
    loss_fn = create_loss(config)
    trainer = Trainer(model, loss_fn, config, dataloaders, tracker)

    # Resume if checkpoint exists
    if resume_ckpt and resume_ckpt.exists():
        trainer.load_checkpoint(str(resume_ckpt))

    # Train
    best_dice = trainer.train()

    # Save fold summary
    best_ckpt_path = tracker.run_dir / "best_model.pth"
    region_dice = {}
    if best_ckpt_path.exists():
        ckpt = torch.load(str(best_ckpt_path), map_location="cpu", weights_only=False)
        region_dice = ckpt.get("region_dice", {})

    summary = {
        "fold": fold_idx,
        "model": config["model"]["name"],
        "best_mean_region_dice": best_dice,
        "best_dice_ET": region_dice.get("ET", None),
        "best_dice_TC": region_dice.get("TC", None),
        "best_dice_WT": region_dice.get("WT", None),
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
    }
    tracker.save_summary(summary)
    tracker.close()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train BraTS segmentation with K-fold CV")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--fold", type=int, default=None,
                        help="Train only this fold (0-indexed). Omit to train all folds.")
    parser.add_argument("--data_dir", type=str, help="Override data.train_dir in config")
    parser.add_argument("--resume_dir", type=str,
                        help="Directory containing fold*/best_model.pth to resume from")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick sanity check: 3 epochs, single fold. Use for code/config validation.")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override training.epochs from config (handy for local timing runs).")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["data"]["train_dir"] = args.data_dir
    if args.smoke_test:
        config["training"]["epochs"] = 3
        config["training"]["val_interval"] = 2
        config["training"]["early_stopping_patience"] = 99
        args.fold = 0 if args.fold is None else args.fold
    if args.max_epochs:
        config["training"]["epochs"] = args.max_epochs

    n_folds = config["data"].get("n_folds", 5)
    config["data"]["n_folds"] = n_folds

    console.print(Panel.fit(
        f"[bold cyan]BraTS {n_folds}-Fold Cross-Validation[/bold cyan]\n"
        f"[dim]Model: {config['model']['name']} | "
        f"Patch: {config['preprocessing']['spatial_size']} | "
        f"Epochs: {config['training']['epochs']}[/dim]",
        border_style="bright_blue"
    ))

    # Validate data dir
    data_dir = Path(config["data"]["train_dir"]).expanduser()
    if not data_dir.exists():
        console.print(f"[red bold]Error: Data directory not found: {data_dir}[/red bold]")
        sys.exit(1)

    # Create K-fold splits (patient-level)
    folds = create_kfold_splits(str(data_dir), n_folds=n_folds, seed=config["data"]["split_seed"])

    # Determine which folds to train
    if args.fold is not None:
        fold_indices = [args.fold]
    else:
        fold_indices = list(range(n_folds))

    # Train each fold
    fold_results = []
    for fold_idx in fold_indices:
        train_cases, val_cases = folds[fold_idx]
        summary = train_fold(config, fold_idx, train_cases, val_cases, args)
        fold_results.append(summary)

    # Print cross-fold summary
    if len(fold_results) > 1:
        console.print("\n")
        table = Table(title=f"{n_folds}-Fold CV Results", style="bold green")
        table.add_column("Fold", style="bold")
        table.add_column("Mean Region", justify="right")
        table.add_column("Dice ET", justify="right")
        table.add_column("Dice TC", justify="right")
        table.add_column("Dice WT", justify="right")

        et_scores, tc_scores, wt_scores, mean_scores = [], [], [], []
        for r in fold_results:
            et = r.get("best_dice_ET", 0) or 0
            tc = r.get("best_dice_TC", 0) or 0
            wt = r.get("best_dice_WT", 0) or 0
            m = r.get("best_mean_region_dice", 0) or 0
            et_scores.append(et)
            tc_scores.append(tc)
            wt_scores.append(wt)
            mean_scores.append(m)
            table.add_row(f"Fold {r['fold']}", f"{m:.4f}", f"{et:.4f}", f"{tc:.4f}", f"{wt:.4f}")

        import numpy as np
        table.add_row("", "", "", "", "")
        table.add_row(
            "[bold]Mean ± Std[/bold]",
            f"{np.mean(mean_scores):.4f} ± {np.std(mean_scores):.4f}",
            f"{np.mean(et_scores):.4f} ± {np.std(et_scores):.4f}",
            f"{np.mean(tc_scores):.4f} ± {np.std(tc_scores):.4f}",
            f"{np.mean(wt_scores):.4f} ± {np.std(wt_scores):.4f}",
        )
        console.print(table)

        # Save aggregated results
        output_dir = Path(config["experiment"]["output_dir"]).expanduser()
        agg_path = output_dir / f"kfold_{config['model']['name']}_summary.json"
        agg = {
            "model": config["model"]["name"],
            "n_folds": n_folds,
            "spatial_size": config["preprocessing"]["spatial_size"],
            "mean_region_dice": f"{np.mean(mean_scores):.4f} ± {np.std(mean_scores):.4f}",
            "dice_ET": f"{np.mean(et_scores):.4f} ± {np.std(et_scores):.4f}",
            "dice_TC": f"{np.mean(tc_scores):.4f} ± {np.std(tc_scores):.4f}",
            "dice_WT": f"{np.mean(wt_scores):.4f} ± {np.std(wt_scores):.4f}",
            "folds": fold_results,
        }
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2, default=str)
        console.print(f"\n[bold green]Aggregated results saved to {agg_path}[/bold green]")


if __name__ == "__main__":
    main()
