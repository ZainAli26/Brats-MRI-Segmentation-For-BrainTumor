#!/usr/bin/env python3
"""BraTS Segmentation - Evaluation & Failure Analysis Entrypoint.

Usage:
    python evaluate.py --run_dir runs/segresnet_20240101_120000
    python evaluate.py --run_dir runs/segresnet_20240101_120000 --split test
    python evaluate.py --run_dir runs/segresnet_20240101_120000 --visualize_failures
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from src.utils.experiment import load_config, ExperimentTracker
from src.data.splits import create_patient_splits
from src.data.dataset import build_file_list
from src.data.preprocessing import get_val_transforms
from src.models.factory import create_model
from src.evaluation.metrics import compute_case_metrics, print_metrics_summary
from src.evaluation.failure_analysis import identify_failure_cases, print_failure_summary, generate_failure_report
from src.evaluation.visualization import (
    visualize_case_comparison,
    visualize_failure_grid,
    plot_metrics_distributions,
    plot_training_curves,
)

from monai.data import CacheDataset, DataLoader

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Evaluate BraTS segmentation model")
    parser.add_argument("--run_dir", required=True, help="Path to training run directory")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--visualize_failures", action="store_true",
                        help="Generate visual overlays for failure cases")
    parser.add_argument("--compare", nargs="+",
                        help="Additional run dirs to compare against")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.exists():
        console.print(f"[red bold]Run directory not found: {run_dir}[/red bold]")
        sys.exit(1)

    # Load config from run
    config = load_config(str(run_dir / "config.yaml"))

    console.print(Panel.fit(
        f"[bold cyan]BraTS Segmentation Evaluation[/bold cyan]\n"
        f"[dim]Model: {config['model']['name']} | Split: {args.split} | Run: {run_dir.name}[/dim]",
        border_style="bright_blue"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[dim]Device: {device}[/dim]\n")

    # Recreate same patient splits
    data_dir = Path(config["data"]["train_dir"]).expanduser()
    train_cases, val_cases, test_cases = create_patient_splits(
        str(data_dir),
        split_ratios=config["data"]["split_ratios"],
        seed=config["data"]["split_seed"],
    )

    eval_cases = test_cases if args.split == "test" else val_cases

    # Build dataloader
    modalities = config["data"]["modalities"]
    label_map = {int(k): int(v) for k, v in config["data"]["label_map"].items()}
    spatial_size = config["preprocessing"]["spatial_size"]

    val_transform = get_val_transforms(spatial_size, modalities, label_map)
    file_list = build_file_list(eval_cases, modalities, include_label=True)

    eval_ds = CacheDataset(file_list, transform=val_transform, cache_rate=1.0, num_workers=2)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)

    # Load model
    console.print("[bold]Loading model...[/bold]")
    model = create_model(config)
    ckpt_path = run_dir / "best_model.pth"
    if not ckpt_path.exists():
        # Try checkpoints subdir
        ckpt_path = run_dir / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        console.print(f"[red]No checkpoint found in {run_dir}[/red]")
        sys.exit(1)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    console.print(f"[green]Loaded checkpoint: epoch {ckpt['epoch']}, val dice {ckpt['val_dice']:.4f}[/green]\n")

    # Compute per-case metrics
    console.print("[bold]Computing per-case metrics...[/bold]")
    metrics_df = compute_case_metrics(model, eval_loader, config, device)

    # Save metrics
    eval_dir = run_dir / f"eval_{args.split}"
    eval_dir.mkdir(exist_ok=True)
    metrics_df.to_csv(eval_dir / "case_metrics.csv", index=False)

    # Print summary
    print_metrics_summary(metrics_df, config)

    # Failure analysis
    console.print("\n[bold]Running failure analysis...[/bold]")
    generate_failure_report(metrics_df, config, str(eval_dir))

    # Visualizations
    console.print("\n[bold]Generating visualizations...[/bold]")
    plot_metrics_distributions(metrics_df, config, str(eval_dir))
    plot_training_curves(str(run_dir / "logs"), str(eval_dir))

    # Visualize failure cases
    if args.visualize_failures:
        console.print("\n[bold]Generating failure case overlays...[/bold]")
        failures = identify_failure_cases(metrics_df, config)

        # Collect all failure case IDs
        failure_ids = set()
        for category, df in failures.items():
            n = config["evaluation"].get("num_failure_cases", 10)
            failure_ids.update(df.head(n)["case_id"].tolist())

        if failure_ids:
            visualize_failure_grid(
                model, eval_loader, list(failure_ids),
                config, str(eval_dir), device,
                max_cases=min(len(failure_ids), 8),
            )

            # Individual case comparisons for top failures
            for batch_data in eval_loader:
                case_id = batch_data.get("case_id", ["unknown"])[0]
                if case_id not in failure_ids:
                    continue

                images = batch_data["image"].to(device)
                labels = batch_data["label"]

                with torch.no_grad():
                    from monai.inferers import sliding_window_inference
                    from monai.transforms import AsDiscrete
                    from torch.cuda.amp import autocast

                    with autocast(enabled=config["training"]["amp"] and device.type == "cuda"):
                        outputs = sliding_window_inference(
                            images, spatial_size, config["training"]["sw_batch_size"],
                            model, overlap=0.5
                        )
                    pred = AsDiscrete(argmax=True)(outputs[0]).cpu().numpy()

                image_np = images[0].cpu().numpy()
                label_np = labels[0].cpu().numpy()
                if label_np.ndim == 4:
                    label_np = label_np[0]
                if pred.ndim == 4:
                    pred = pred[0]

                # Get metrics for this case
                case_metrics = {}
                row = metrics_df[metrics_df["case_id"] == case_id]
                if not row.empty:
                    for col in row.columns:
                        if col.startswith("dice_"):
                            case_metrics[col] = row.iloc[0][col]

                visualize_case_comparison(
                    image_np, label_np, pred, case_id,
                    str(eval_dir / "case_visualizations"),
                    metrics=case_metrics,
                )
        else:
            console.print("[green]No failure cases identified![/green]")

    # Model comparison
    if args.compare:
        console.print("\n[bold]Comparing models...[/bold]")
        results = {run_dir.name: metrics_df}
        for other_dir in args.compare:
            other_path = Path(other_dir).expanduser()
            other_csv = other_path / f"eval_{args.split}" / "case_metrics.csv"
            if other_csv.exists():
                results[other_path.name] = pd.read_csv(other_csv)
            else:
                console.print(f"[yellow]No metrics found for {other_dir}, skipping[/yellow]")

        if len(results) > 1:
            from src.evaluation.visualization import plot_model_comparison
            plot_model_comparison(results, str(eval_dir))

    console.print(f"\n[bold green]Evaluation complete. Results in: {eval_dir}[/bold green]")


if __name__ == "__main__":
    main()
