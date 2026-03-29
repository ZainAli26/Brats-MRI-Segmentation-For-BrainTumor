"""Failure case analysis for BraTS segmentation.

Identifies cases where models perform poorly, especially for
small or fragmented enhancing tumor (ET) regions.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def identify_failure_cases(
    metrics_df: pd.DataFrame,
    config: dict,
) -> Dict[str, pd.DataFrame]:
    """Identify different types of failure cases.

    Returns dict of DataFrames:
        - "low_dice": Cases below dice threshold
        - "small_tumor_failures": Small tumors with low performance
        - "et_failures": ET-specific failures
        - "fragmented": Cases with fragmented predictions
    """
    threshold = config["evaluation"]["failure_dice_threshold"]
    small_vol = config["evaluation"]["small_tumor_volume_threshold"]
    regions = config["evaluation"]["regions"]

    failures = {}

    # 1. Overall low-Dice cases
    dice_cols = [c for c in metrics_df.columns if c.startswith("dice_") and c.split("_")[1] in regions]
    if dice_cols:
        metrics_df["mean_region_dice"] = metrics_df[dice_cols].mean(axis=1)
        low_dice = metrics_df[metrics_df["mean_region_dice"] < threshold].sort_values("mean_region_dice")
        failures["low_dice"] = low_dice

    # 2. ET-specific failures
    if "dice_ET" in metrics_df.columns:
        et_failures = metrics_df[metrics_df["dice_ET"] < threshold].sort_values("dice_ET")
        failures["et_failures"] = et_failures

    # 3. Small tumor failures
    if "vol_true_ET" in metrics_df.columns:
        small_mask = (metrics_df["vol_true_ET"] > 0) & (metrics_df["vol_true_ET"] < small_vol)
        small_tumors = metrics_df[small_mask].copy()
        if not small_tumors.empty:
            small_tumors = small_tumors.sort_values("dice_ET")
            failures["small_tumor_failures"] = small_tumors

    # 4. Cases with over-segmentation (pred >> true volume)
    if "vol_pred_WT" in metrics_df.columns and "vol_true_WT" in metrics_df.columns:
        metrics_df["vol_ratio_WT"] = metrics_df["vol_pred_WT"] / (metrics_df["vol_true_WT"] + 1)
        overseg = metrics_df[metrics_df["vol_ratio_WT"] > 2.0].sort_values("vol_ratio_WT", ascending=False)
        if not overseg.empty:
            failures["oversegmentation"] = overseg

    # 5. Cases with no ET predicted but ET present in ground truth
    if "vol_pred_ET" in metrics_df.columns and "vol_true_ET" in metrics_df.columns:
        missed_et = metrics_df[
            (metrics_df["vol_true_ET"] > 0) & (metrics_df["vol_pred_ET"] == 0)
        ]
        if not missed_et.empty:
            failures["missed_et"] = missed_et

    return failures


def print_failure_summary(failures: Dict[str, pd.DataFrame], n_show: int = 10):
    """Print summary of failure cases."""
    for category, df in failures.items():
        if df.empty:
            continue

        title = {
            "low_dice": "Low Overall Dice Cases",
            "et_failures": "ET-Specific Failures",
            "small_tumor_failures": "Small Tumor Failures",
            "oversegmentation": "Over-Segmentation Cases",
            "missed_et": "Missed ET Cases (False Negatives)",
        }.get(category, category)

        table = Table(title=f"{title} ({len(df)} cases)", style="bold red")
        table.add_column("Case ID", style="bold")

        # Add relevant metric columns
        show_cols = [c for c in df.columns if c.startswith("dice_") or c.startswith("vol_")]
        for col in show_cols[:6]:  # Limit columns
            table.add_column(col, justify="right")

        for _, row in df.head(n_show).iterrows():
            values = [row["case_id"]] + [
                f"{row[c]:.4f}" if isinstance(row[c], float) else str(int(row[c]))
                for c in show_cols[:6]
            ]
            table.add_row(*values)

        console.print(table)
        console.print()


def generate_failure_report(
    metrics_df: pd.DataFrame,
    config: dict,
    output_dir: str,
) -> Path:
    """Generate a comprehensive failure analysis report saved as CSV."""
    output_path = Path(output_dir)
    failures = identify_failure_cases(metrics_df, config)

    # Save each category
    for category, df in failures.items():
        if not df.empty:
            path = output_path / f"failures_{category}.csv"
            df.to_csv(path, index=False)

    # Print summary
    print_failure_summary(failures, n_show=config["evaluation"].get("num_failure_cases", 10))

    # Summary stats
    console.print("\n[bold]Failure Analysis Summary:[/bold]")
    for category, df in failures.items():
        console.print(f"  {category}: {len(df)} cases")

    return output_path
