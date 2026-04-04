"""Per-class and per-region evaluation metrics for BraTS segmentation."""

from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from src.utils import inference_wrapper
from monai.data import decollate_batch
from torch.cuda.amp import autocast
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()


def compute_case_metrics(
    model,
    dataloader,
    config: dict,
    device: torch.device,
) -> pd.DataFrame:
    """Compute per-case, per-class, and per-region metrics.

    Returns a DataFrame with one row per case containing:
    - Per-class Dice and HD95
    - Per-region (ET, TC, WT) Dice and HD95
    - Tumor volume info
    """
    model.eval()
    num_classes = config["data"]["num_classes"]
    spatial_size = config["preprocessing"]["spatial_size"]
    sw_batch = config["training"]["sw_batch_size"]
    sw_overlap = config["training"]["sw_overlap"]
    regions = config["evaluation"]["regions"]
    use_amp = config["training"]["amp"] and device.type == "cuda"

    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

    inverse_label_map = config["data"]["inverse_label_map"]
    class_names = {1: "NCR", 2: "ED", 3: "ET"}

    records = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating cases"):
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            case_id = batch_data.get("case_id", ["unknown"])[0]

            with autocast(enabled=use_amp):
                outputs = sliding_window_inference(
                    images, spatial_size, sw_batch, inference_wrapper(model), overlap=sw_overlap
                )

            outputs_list = decollate_batch(outputs)
            labels_list = decollate_batch(labels)

            pred_oh = post_pred(outputs_list[0])
            lab_oh = post_label(labels_list[0])

            # Per-class Dice
            dice_metric.reset()
            dice_metric(y_pred=[pred_oh], y=[lab_oh])
            class_dice = dice_metric.aggregate().cpu().numpy().flatten()

            # Per-class HD95
            hd_metric.reset()
            try:
                hd_metric(y_pred=[pred_oh], y=[lab_oh])
                class_hd = hd_metric.aggregate().cpu().numpy().flatten()
            except Exception:
                class_hd = [np.nan] * (num_classes - 1)

            record = {"case_id": case_id}

            # Per-class metrics
            for i, cname in class_names.items():
                idx = i - 1  # 0-indexed for metrics (background excluded)
                record[f"dice_{cname}"] = float(class_dice[idx]) if idx < len(class_dice) else np.nan
                record[f"hd95_{cname}"] = float(class_hd[idx]) if idx < len(class_hd) else np.nan

            # Per-region metrics
            pred_argmax = pred_oh.argmax(dim=0)
            lab_argmax = lab_oh.argmax(dim=0)

            for region_name, label_indices in regions.items():
                pred_region = torch.zeros_like(pred_argmax, dtype=torch.bool)
                lab_region = torch.zeros_like(lab_argmax, dtype=torch.bool)
                for idx in label_indices:
                    pred_region |= (pred_argmax == idx)
                    lab_region |= (lab_argmax == idx)

                # Dice
                intersection = (pred_region & lab_region).sum().float()
                union = pred_region.sum().float() + lab_region.sum().float()
                region_dice = (2.0 * intersection / (union + 1e-7)).item()
                record[f"dice_{region_name}"] = region_dice

                # Volume info
                record[f"vol_pred_{region_name}"] = int(pred_region.sum().item())
                record[f"vol_true_{region_name}"] = int(lab_region.sum().item())

            records.append(record)

    df = pd.DataFrame(records)
    return df


def print_metrics_summary(df: pd.DataFrame, config: dict):
    """Print a rich summary table of evaluation metrics."""
    regions = config["evaluation"]["regions"]
    class_names = ["NCR", "ED", "ET"]

    # Per-class table
    table = Table(title="Per-Class Metrics", style="bold cyan")
    table.add_column("Class", style="bold")
    table.add_column("Mean Dice", justify="right")
    table.add_column("Std Dice", justify="right")
    table.add_column("Mean HD95", justify="right")

    for cname in class_names:
        col = f"dice_{cname}"
        hd_col = f"hd95_{cname}"
        if col in df.columns:
            mean_d = df[col].mean()
            std_d = df[col].std()
            mean_hd = df[hd_col].mean() if hd_col in df.columns else np.nan
            table.add_row(cname, f"{mean_d:.4f}", f"{std_d:.4f}", f"{mean_hd:.2f}")

    console.print(table)

    # Per-region table
    table2 = Table(title="Per-Region Metrics (BraTS Standard)", style="bold magenta")
    table2.add_column("Region", style="bold")
    table2.add_column("Mean Dice", justify="right")
    table2.add_column("Std Dice", justify="right")
    table2.add_column("Median Dice", justify="right")

    for region_name in regions:
        col = f"dice_{region_name}"
        if col in df.columns:
            table2.add_row(
                region_name,
                f"{df[col].mean():.4f}",
                f"{df[col].std():.4f}",
                f"{df[col].median():.4f}",
            )

    console.print(table2)
