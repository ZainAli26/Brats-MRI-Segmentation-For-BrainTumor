#!/usr/bin/env python3
"""BraTS Segmentation - Training Entrypoint.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --model dynunet
    python train.py --config configs/config.yaml --model swin_unetr --epochs 200 --batch_size 1
"""

import argparse
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel

from src.utils.experiment import load_config, ExperimentTracker
from src.data.splits import create_patient_splits
from src.data.dataset import get_dataloaders
from src.data.preprocessing import get_train_transforms, get_val_transforms
from src.models.factory import create_model
from src.training.losses import create_loss
from src.training.trainer import Trainer

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train BraTS segmentation model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--model", choices=["nnunet_v2", "dynunet", "swin_unetr", "segresnet"],
                        help="Override model in config")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.model:
        config["model"]["name"] = args.model
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    # Banner
    console.print(Panel.fit(
        f"[bold cyan]BraTS Segmentation Training[/bold cyan]\n"
        f"[dim]Model: {config['model']['name']} | "
        f"Epochs: {config['training']['epochs']} | "
        f"Batch: {config['training']['batch_size']} | "
        f"LR: {config['training']['learning_rate']}[/dim]",
        border_style="bright_blue"
    ))

    # Validate data dir
    data_dir = Path(config["data"]["train_dir"]).expanduser()
    if not data_dir.exists():
        console.print(f"[red bold]Error: Data directory not found: {data_dir}[/red bold]")
        sys.exit(1)

    # Initialize experiment tracker
    tracker = ExperimentTracker(config, config_path=args.config)

    # Patient-level splits
    console.print("\n[bold]Creating patient-level data splits...[/bold]")
    train_cases, val_cases, test_cases = create_patient_splits(
        str(data_dir),
        split_ratios=config["data"]["split_ratios"],
        seed=config["data"]["split_seed"],
    )

    # Build transforms (identical preprocessing for all models)
    console.print("\n[bold]Building preprocessing pipeline...[/bold]")
    modalities = config["data"]["modalities"]
    label_map = {int(k): int(v) for k, v in config["data"]["label_map"].items()}
    spatial_size = config["preprocessing"]["spatial_size"]
    aug_config = config["preprocessing"]["augmentation"]

    train_transform = get_train_transforms(spatial_size, modalities, label_map, aug_config)
    val_transform = get_val_transforms(spatial_size, modalities, label_map)

    # Create dataloaders
    console.print("[bold]Creating dataloaders...[/bold]")
    dataloaders = get_dataloaders(
        train_cases, val_cases, test_cases,
        modalities=modalities,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    console.print(f"  Train: {len(dataloaders['train'].dataset)} samples")
    console.print(f"  Val:   {len(dataloaders['val'].dataset)} samples")
    console.print(f"  Test:  {len(dataloaders['test'].dataset)} samples")

    # Create model
    console.print("\n[bold]Creating model...[/bold]")
    model = create_model(config)

    # Create loss
    loss_fn = create_loss(config)

    # Create trainer
    trainer = Trainer(model, loss_fn, config, dataloaders, tracker)

    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    console.print()
    best_dice = trainer.train()

    # Save final summary
    tracker.save_summary({
        "model": config["model"]["name"],
        "best_val_dice": best_dice,
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "test_cases": len(test_cases),
    })

    tracker.close()
    console.print(f"\n[bold green]Run saved to: {tracker.run_dir}[/bold green]")


if __name__ == "__main__":
    main()
