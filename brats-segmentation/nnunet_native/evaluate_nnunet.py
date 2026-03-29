#!/usr/bin/env python3
"""Evaluate native nnU-Net v2 predictions using our shared metrics & visualization.

Bridges nnU-Net's prediction output back into our evaluation pipeline so that
native nnU-Net results are directly comparable with our custom-loop models
(nnunet_v2, SwinUNETR, SegResNet).

nnU-Net outputs predictions as single-channel NIfTI with remapped labels (0,1,2,3).
We load these alongside the ground truth and compute the same per-case, per-class,
and per-region metrics used by evaluate.py.

Usage:
    python nnunet_native/evaluate_nnunet.py \
        --pred_dir nnunet_data/nnUNet_results/.../test_predictions \
        --data_dir ../Brats2024/training_data1_v2 \
        --output_dir runs/nnunet_native_eval

    # Compare with custom-loop runs
    python analyze_failures.py \
        --run_dirs runs/nnunet_native_eval runs/segresnet_20240101_120000 \
        --compare
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.splits import create_patient_splits
from src.evaluation.failure_analysis import identify_failure_cases, print_failure_summary
from src.evaluation.visualization import (
    visualize_case_comparison,
    plot_metrics_distributions,
)
from src.utils.experiment import load_config

console = Console()

# Same label remap used during conversion
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}

# Evaluation regions (using remapped labels)
REGIONS = {
    "ET": [3],
    "TC": [1, 3],
    "WT": [1, 2, 3],
}
CLASS_NAMES = {1: "NCR", 2: "ED", 3: "ET"}


def _dice(pred_mask, true_mask):
    intersection = np.sum(pred_mask & true_mask)
    total = np.sum(pred_mask) + np.sum(true_mask)
    if total == 0:
        return 1.0 if np.sum(true_mask) == 0 else 0.0
    return 2.0 * intersection / total


def _hausdorff95(pred_mask, true_mask):
    """Compute 95th percentile Hausdorff distance."""
    if not np.any(pred_mask) or not not np.any(true_mask):
        if not np.any(pred_mask) and not np.any(true_mask):
            return 0.0
        return np.nan

    from scipy.ndimage import distance_transform_edt
    pred_surface = pred_mask ^ ndimage.binary_erosion(pred_mask)
    true_surface = true_mask ^ ndimage.binary_erosion(true_mask)

    if not np.any(pred_surface) or not np.any(true_surface):
        return np.nan

    dt_pred = distance_transform_edt(~true_surface)
    dt_true = distance_transform_edt(~pred_surface)

    d_pred_to_true = dt_pred[pred_surface]
    d_true_to_pred = dt_true[true_surface]

    all_distances = np.concatenate([d_pred_to_true, d_true_to_pred])
    return float(np.percentile(all_distances, 95))


def evaluate_predictions(
    pred_dir: str,
    data_dir: str,
    output_dir: str,
    split_ratios: list = [0.75, 0.15, 0.10],
    split_seed: int = 42,
    visualize: bool = True,
    config_path: str = None,
):
    pred_path = Path(pred_dir).expanduser().resolve()
    data_path = Path(data_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold cyan]nnU-Net v2 Native Evaluation[/bold cyan]\n"
        f"[dim]Predictions: {pred_path}\nGround truth: {data_path}[/dim]",
        border_style="bright_blue"
    ))

    # Get test split (same as used in conversion)
    _, _, test_cases = create_patient_splits(str(data_path), split_ratios, split_seed)
    test_ids = {c.name for c in test_cases}

    # Find prediction files
    pred_files = sorted(pred_path.glob("*.nii.gz"))
    if not pred_files:
        console.print(f"[red]No prediction files found in {pred_path}[/red]")
        return

    console.print(f"\n[bold]Evaluating {len(pred_files)} predictions...[/bold]")

    records = []
    vis_data = []

    for pred_file in tqdm(pred_files, desc="Computing metrics"):
        case_id = pred_file.stem.replace(".nii", "")  # handle .nii.gz

        # Load prediction (already remapped labels: 0,1,2,3)
        pred_img = nib.load(str(pred_file))
        pred_data = pred_img.get_fdata().astype(np.uint8)

        # Load ground truth
        case_dir = data_path / case_id
        if not case_dir.exists():
            console.print(f"[yellow]Ground truth dir not found for {case_id}, skipping[/yellow]")
            continue

        seg_files = list(case_dir.glob("*-seg.nii.gz"))
        if not seg_files:
            console.print(f"[yellow]No seg file for {case_id}, skipping[/yellow]")
            continue

        gt_img = nib.load(str(seg_files[0]))
        gt_data = gt_img.get_fdata().astype(np.uint8)

        # Remap ground truth labels
        gt_remapped = np.zeros_like(gt_data)
        for src, dst in LABEL_REMAP.items():
            gt_remapped[gt_data == src] = dst

        record = {"case_id": case_id}

        # Per-class metrics
        for class_idx, class_name in CLASS_NAMES.items():
            pred_mask = pred_data == class_idx
            true_mask = gt_remapped == class_idx
            record[f"dice_{class_name}"] = _dice(pred_mask, true_mask)
            try:
                record[f"hd95_{class_name}"] = _hausdorff95(pred_mask, true_mask)
            except Exception:
                record[f"hd95_{class_name}"] = np.nan

        # Per-region metrics
        for region_name, label_indices in REGIONS.items():
            pred_region = np.isin(pred_data, label_indices)
            true_region = np.isin(gt_remapped, label_indices)
            record[f"dice_{region_name}"] = _dice(pred_region, true_region)
            record[f"vol_pred_{region_name}"] = int(pred_region.sum())
            record[f"vol_true_{region_name}"] = int(true_region.sum())

        records.append(record)

        # Save data for visualization of worst cases
        if visualize:
            vis_data.append((case_id, case_dir, pred_data, gt_remapped, record.copy()))

    metrics_df = pd.DataFrame(records)

    # Save metrics
    metrics_df.to_csv(output_path / "case_metrics.csv", index=False)
    console.print(f"[green]Saved metrics to {output_path / 'case_metrics.csv'}[/green]")

    # Print summary
    from rich.table import Table

    table = Table(title="nnU-Net v2 Native - Per-Region Metrics", style="bold magenta")
    table.add_column("Region", style="bold")
    table.add_column("Mean Dice", justify="right")
    table.add_column("Std Dice", justify="right")
    table.add_column("Median Dice", justify="right")

    for region in REGIONS:
        col = f"dice_{region}"
        if col in metrics_df.columns:
            table.add_row(
                region,
                f"{metrics_df[col].mean():.4f}",
                f"{metrics_df[col].std():.4f}",
                f"{metrics_df[col].median():.4f}",
            )
    console.print(table)

    table2 = Table(title="nnU-Net v2 Native - Per-Class Metrics", style="bold cyan")
    table2.add_column("Class", style="bold")
    table2.add_column("Mean Dice", justify="right")
    table2.add_column("Mean HD95", justify="right")

    for cname in CLASS_NAMES.values():
        d_col = f"dice_{cname}"
        h_col = f"hd95_{cname}"
        if d_col in metrics_df.columns:
            table2.add_row(cname, f"{metrics_df[d_col].mean():.4f}",
                           f"{metrics_df[h_col].mean():.2f}" if h_col in metrics_df.columns else "N/A")
    console.print(table2)

    # Failure analysis using our shared code
    eval_config = {
        "evaluation": {
            "regions": REGIONS,
            "failure_dice_threshold": 0.5,
            "small_tumor_volume_threshold": 500,
            "num_failure_cases": 10,
        }
    }

    console.print("\n[bold]Failure Analysis:[/bold]")
    failures = identify_failure_cases(metrics_df, eval_config)
    print_failure_summary(failures)

    # Visualizations
    if visualize:
        plot_metrics_distributions(metrics_df, eval_config, str(output_path))

        # Visualize worst ET cases
        if "dice_ET" in metrics_df.columns:
            worst_et = metrics_df.nsmallest(5, "dice_ET")["case_id"].tolist()
            viz_dir = output_path / "case_visualizations"
            viz_dir.mkdir(exist_ok=True)

            for case_id, case_dir, pred_data, gt_data, case_metrics in vis_data:
                if case_id not in worst_et:
                    continue

                # Load one modality for visualization
                t1c_files = list(case_dir.glob("*-t1c.nii.gz"))
                if not t1c_files:
                    continue
                mri = nib.load(str(t1c_files[0])).get_fdata()
                # Stack as fake 4-channel (just use t1c)
                mri_4ch = np.stack([mri] * 4, axis=0)

                path = visualize_case_comparison(
                    mri_4ch, gt_data, pred_data, case_id,
                    str(viz_dir),
                    metrics={k: v for k, v in case_metrics.items() if k.startswith("dice_") and isinstance(v, float)},
                )
                console.print(f"  Saved: {path}")

    # Save a fake config.yaml so analyze_failures.py can load this run
    import yaml
    fake_config = {
        "model": {"name": "nnunet_v2_native"},
        "data": {"train_dir": str(data_path), "split_ratios": split_ratios, "split_seed": split_seed},
        "evaluation": eval_config["evaluation"],
    }
    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(fake_config, f)

    console.print(f"\n[bold green]Evaluation complete. Results in: {output_path}[/bold green]")
    console.print(f"[dim]Compare with custom-loop models:[/dim]")
    console.print(f"  python analyze_failures.py --run_dirs {output_path} runs/<custom_run> --compare")

    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate native nnU-Net v2 predictions")
    parser.add_argument("--pred_dir", required=True, help="Directory with nnU-Net prediction NIfTIs")
    parser.add_argument("--data_dir", default="../Brats2024/training_data1_v2", help="Original BraTS data directory")
    parser.add_argument("--output_dir", default="runs/nnunet_native_eval", help="Output directory for results")
    parser.add_argument("--config", help="Optional: path to project config.yaml for split params")
    parser.add_argument("--no_visualize", action="store_true", help="Skip generating visualizations")
    args = parser.parse_args()

    split_ratios = [0.75, 0.15, 0.10]
    split_seed = 42
    if args.config:
        config = load_config(args.config)
        split_ratios = config["data"]["split_ratios"]
        split_seed = config["data"]["split_seed"]

    evaluate_predictions(
        args.pred_dir, args.data_dir, args.output_dir,
        split_ratios, split_seed,
        visualize=not args.no_visualize,
    )
