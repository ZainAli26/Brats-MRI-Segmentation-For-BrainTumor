#!/usr/bin/env python3
"""Standalone failure analysis and visual debugging tool.

Usage:
    # Analyze a specific run's results
    python analyze_failures.py --run_dir runs/segresnet_20240101_120000

    # Compare failure patterns across models
    python analyze_failures.py --run_dirs runs/segresnet_* --compare

    # Visualize specific problematic cases
    python analyze_failures.py --run_dir runs/segresnet_20240101_120000 --cases BraTS-GLI-00463-100 BraTS-GLI-00528-101
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.experiment import load_config
from src.data.splits import create_patient_splits
from src.data.dataset import build_file_list
from src.data.preprocessing import get_val_transforms
from src.models.factory import create_model
from src.evaluation.failure_analysis import identify_failure_cases, print_failure_summary
from src.evaluation.visualization import (
    visualize_case_comparison,
    visualize_failure_grid,
    plot_metrics_distributions,
    plot_model_comparison,
)
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from torch.cuda.amp import autocast

console = Console()


def analyze_single_run(run_dir: Path, split: str = "test", case_ids: list = None):
    """Deep-dive failure analysis for a single run."""
    config = load_config(str(run_dir / "config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check for pre-computed metrics
    eval_dir = run_dir / f"eval_{split}"
    metrics_path = eval_dir / "case_metrics.csv"

    if metrics_path.exists():
        console.print(f"[dim]Loading pre-computed metrics from {metrics_path}[/dim]")
        metrics_df = pd.read_csv(metrics_path)
    else:
        console.print("[yellow]No pre-computed metrics found. Run evaluate.py first.[/yellow]")
        sys.exit(1)

    # Print overall stats
    console.print(Panel.fit(
        f"[bold cyan]Failure Analysis: {run_dir.name}[/bold cyan]\n"
        f"[dim]Split: {split} | Cases: {len(metrics_df)}[/dim]",
        border_style="bright_blue"
    ))

    # Identify failures
    failures = identify_failure_cases(metrics_df, config)
    print_failure_summary(failures, n_show=config["evaluation"].get("num_failure_cases", 10))

    # Detailed stats for small tumors
    if "vol_true_ET" in metrics_df.columns:
        console.print("\n[bold]Small Tumor Analysis:[/bold]")
        vol_thresh = config["evaluation"]["small_tumor_volume_threshold"]
        small = metrics_df[
            (metrics_df["vol_true_ET"] > 0) & (metrics_df["vol_true_ET"] < vol_thresh)
        ]
        large = metrics_df[metrics_df["vol_true_ET"] >= vol_thresh]
        no_et = metrics_df[metrics_df["vol_true_ET"] == 0]

        table = Table(title="ET Performance by Tumor Size")
        table.add_column("Category", style="bold")
        table.add_column("N Cases", justify="right")
        table.add_column("Mean ET Dice", justify="right")
        table.add_column("Mean WT Dice", justify="right")

        if "dice_ET" in metrics_df.columns:
            if not no_et.empty:
                table.add_row("No ET present", str(len(no_et)), "N/A",
                              f"{no_et['dice_WT'].mean():.4f}" if "dice_WT" in no_et.columns else "N/A")
            if not small.empty:
                table.add_row(f"Small ET (<{vol_thresh} vox)", str(len(small)),
                              f"{small['dice_ET'].mean():.4f}",
                              f"{small['dice_WT'].mean():.4f}" if "dice_WT" in small.columns else "N/A")
            if not large.empty:
                table.add_row(f"Large ET (>={vol_thresh} vox)", str(len(large)),
                              f"{large['dice_ET'].mean():.4f}",
                              f"{large['dice_WT'].mean():.4f}" if "dice_WT" in large.columns else "N/A")

        console.print(table)

    # Generate visualizations
    viz_dir = eval_dir / "failure_analysis"
    viz_dir.mkdir(exist_ok=True)
    plot_metrics_distributions(metrics_df, config, str(viz_dir))

    # If specific cases requested, generate detailed views
    if case_ids:
        console.print(f"\n[bold]Generating detailed views for {len(case_ids)} cases...[/bold]")

        data_dir = Path(config["data"]["train_dir"]).expanduser()
        train_cases, val_cases, test_cases = create_patient_splits(
            str(data_dir),
            split_ratios=config["data"]["split_ratios"],
            seed=config["data"]["split_seed"],
        )
        eval_cases = test_cases if split == "test" else val_cases

        modalities = config["data"]["modalities"]
        label_map = {int(k): int(v) for k, v in config["data"]["label_map"].items()}
        spatial_size = config["preprocessing"]["spatial_size"]

        val_transform = get_val_transforms(spatial_size, modalities, label_map)
        file_list = build_file_list(eval_cases, modalities, include_label=True)
        eval_ds = CacheDataset(file_list, transform=val_transform, cache_rate=1.0, num_workers=2)
        eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)

        model = create_model(config)
        ckpt_path = run_dir / "best_model.pth"
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        post_pred = AsDiscrete(argmax=True)

        with torch.no_grad():
            for batch_data in eval_loader:
                cid = batch_data.get("case_id", ["unknown"])[0]
                if cid not in case_ids:
                    continue

                images = batch_data["image"].to(device)
                labels = batch_data["label"]

                with autocast(enabled=config["training"]["amp"] and device.type == "cuda"):
                    outputs = sliding_window_inference(
                        images, spatial_size,
                        config["training"]["sw_batch_size"],
                        model, overlap=0.5,
                    )

                pred = post_pred(outputs[0]).cpu().numpy()
                image_np = images[0].cpu().numpy()
                label_np = labels[0].cpu().numpy()
                if label_np.ndim == 4:
                    label_np = label_np[0]
                if pred.ndim == 4:
                    pred = pred[0]

                case_metrics = {}
                row = metrics_df[metrics_df["case_id"] == cid]
                if not row.empty:
                    for col in row.columns:
                        if col.startswith("dice_"):
                            case_metrics[col] = row.iloc[0][col]

                path = visualize_case_comparison(
                    image_np, label_np, pred, cid,
                    str(viz_dir / "cases"), metrics=case_metrics,
                )
                console.print(f"  Saved: {path}")

    console.print(f"\n[bold green]Analysis complete. Results in: {viz_dir}[/bold green]")


def compare_runs(run_dirs: list, split: str = "test"):
    """Compare failure patterns across multiple model runs."""
    results = {}
    for rd in run_dirs:
        run_path = Path(rd).expanduser()
        metrics_path = run_path / f"eval_{split}" / "case_metrics.csv"
        if metrics_path.exists():
            results[run_path.name] = pd.read_csv(metrics_path)
            console.print(f"[green]Loaded {run_path.name}[/green]")
        else:
            console.print(f"[yellow]Skipping {rd} - no metrics found[/yellow]")

    if len(results) < 2:
        console.print("[red]Need at least 2 runs to compare[/red]")
        return

    # Summary comparison table
    table = Table(title="Model Comparison Summary")
    table.add_column("Model", style="bold")
    for region in ["ET", "TC", "WT"]:
        table.add_column(f"Dice {region}", justify="right")

    for name, df in results.items():
        row = [name]
        for region in ["ET", "TC", "WT"]:
            col = f"dice_{region}"
            if col in df.columns:
                row.append(f"{df[col].mean():.4f} +/- {df[col].std():.4f}")
            else:
                row.append("N/A")
        table.add_row(*row)

    console.print(table)

    # Generate comparison plots
    output_dir = Path("runs/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(results, str(output_dir))


def main():
    parser = argparse.ArgumentParser(description="BraTS failure analysis and visual debugging")
    parser.add_argument("--run_dir", help="Single run directory to analyze")
    parser.add_argument("--run_dirs", nargs="+", help="Multiple run directories to compare")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--cases", nargs="+", help="Specific case IDs to visualize")
    parser.add_argument("--compare", action="store_true", help="Compare multiple runs")
    args = parser.parse_args()

    if args.compare and args.run_dirs:
        compare_runs(args.run_dirs, args.split)
    elif args.run_dir:
        analyze_single_run(Path(args.run_dir), args.split, args.cases)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
