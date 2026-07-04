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

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
from rich.panel import Panel

from src.utils.experiment import load_config, ExperimentTracker
from src.data.splits import create_patient_splits, resolve_train_dirs
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
    parser.add_argument("--data_dir", type=str, help="Override data.train_dir in config")
    parser.add_argument("--extra_data_dir", action="append", default=None,
                        help="Override data.extra_train_dirs (pooled into the split). "
                             "Repeat for multiple dirs. Use container paths under Docker.")
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
    if args.data_dir:
        config["data"]["train_dir"] = args.data_dir
    if args.extra_data_dir:
        config["data"]["extra_train_dirs"] = args.extra_data_dir

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
    # When resuming, write to the original run directory (derived from checkpoint path)
    resume_dir = None
    if args.resume:
        resume_dir = str(Path(args.resume).expanduser().parent)
    tracker = ExperimentTracker(config, config_path=args.config, resume_dir=resume_dir)

    # Patient-level splits (pool train_dir + any extra_train_dirs first)
    data_sources = resolve_train_dirs(config["data"])
    if len(data_sources) > 1:
        console.print(f"\n[bold]Pooling {len(data_sources)} training sources:[/bold]")
        for d in data_sources:
            console.print(f"  • {d}")
    console.print("\n[bold]Creating patient-level data splits...[/bold]")
    train_cases, val_cases, test_cases = create_patient_splits(
        data_sources,
        split_ratios=config["data"]["split_ratios"],
        seed=config["data"]["split_seed"],
    )

    # Overfit sanity-check mode: train + validate + test on the SAME small fixed
    # subset, so we can confirm a model/pipeline can actually memorize the data
    # (region Dice -> ~1.0). Augmentation and early stopping are disabled so the
    # model is free to fit the patches and runs the full epoch budget.
    overfit_cfg = config["data"].get("overfit") or {}
    if overfit_cfg.get("enabled"):
        n = int(overfit_cfg.get("num_cases", 50))
        subset = sorted(train_cases, key=lambda p: p.name)[:n]
        if len(subset) < n:
            console.print(f"[yellow]Only {len(subset)} cases available for overfit (requested {n})[/yellow]")
        train_cases = val_cases = test_cases = subset
        # Disable augmentation (zero out every probability / magnitude knob).
        for k in list(config["preprocessing"]["augmentation"].keys()):
            config["preprocessing"]["augmentation"][k] = 0
        # Disable early stopping so the full epoch budget runs.
        config["training"]["early_stopping_patience"] = config["training"]["epochs"] + 1
        console.print(Panel.fit(
            f"[bold yellow]OVERFIT MODE[/bold yellow]\n"
            f"[dim]{len(subset)} cases | train = val = test | augmentation OFF | "
            f"early-stopping OFF[/dim]",
            border_style="yellow",
        ))

    # Build transforms (identical preprocessing for all models)
    console.print("\n[bold]Building preprocessing pipeline...[/bold]")
    modalities = config["data"]["modalities"]
    label_map = {int(k): int(v) for k, v in config["data"]["label_map"].items()}
    spatial_size = config["preprocessing"]["spatial_size"]
    aug_config = config["preprocessing"]["augmentation"]

    train_transform = get_train_transforms(spatial_size, modalities, label_map, aug_config)
    val_transform = get_val_transforms(spatial_size, modalities, label_map)

    # Create dataloaders
    cache_dir = config.get("data", {}).get("cache_dir", None)
    if cache_dir:
        console.print(f"[bold]Creating dataloaders with disk cache: {cache_dir}[/bold]")
    else:
        console.print("[bold]Creating dataloaders (no caching — each sample loaded on the fly)...[/bold]")
    dataloaders = get_dataloaders(
        train_cases, val_cases, test_cases,
        modalities=modalities,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        cache_dir=cache_dir,
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

    # Load region dice from best checkpoint for summary
    best_ckpt_path = tracker.run_dir / "best_model.pth"
    region_dice = {}
    if best_ckpt_path.exists():
        import torch
        ckpt = torch.load(str(best_ckpt_path), map_location="cpu", weights_only=False)
        region_dice = ckpt.get("region_dice", {})

    # Save final summary
    tracker.save_summary({
        "model": config["model"]["name"],
        "best_mean_region_dice": best_dice,
        "best_dice_ET": region_dice.get("ET", None),
        "best_dice_TC": region_dice.get("TC", None),
        "best_dice_WT": region_dice.get("WT", None),
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "spatial_size": config["preprocessing"]["spatial_size"],
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "test_cases": len(test_cases),
    })

    tracker.close()
    console.print(f"\n[bold green]Run saved to: {tracker.run_dir}[/bold green]")


if __name__ == "__main__":
    main()
