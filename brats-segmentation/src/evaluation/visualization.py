"""Visualization tools for debugging, result analysis, and failure case inspection.

Generates:
- Per-case overlay visualizations (MRI + segmentation + prediction)
- Failure case comparison grids
- Training curves and metric distribution plots
- Volume scatter plots (predicted vs true)
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import torch
from monai.inferers import sliding_window_inference
from src.utils import inference_wrapper
from monai.transforms import AsDiscrete
from torch.cuda.amp import autocast
from rich.console import Console

console = Console()

# Color map for tumor labels (remapped: 1=NCR, 2=ED, 3=ET)
LABEL_COLORS = {
    1: [1.0, 0.0, 0.0, 0.6],   # NCR - red
    2: [0.0, 1.0, 0.0, 0.6],   # ED - green
    3: [0.0, 0.3, 1.0, 0.6],   # ET - blue
}

DARK_BG = "#1a1a2e"


def _normalize_slice(s: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to [0, 1] using 1-99th percentile of nonzero."""
    nonzero = s[s > 0]
    if len(nonzero) == 0:
        return np.zeros_like(s)
    p1, p99 = np.percentile(nonzero, [1, 99])
    s = np.clip(s, p1, p99)
    if p99 > p1:
        s = (s - p1) / (p99 - p1)
    return s


def _overlay_seg_on_mri(mri_2d: np.ndarray, seg_2d: np.ndarray) -> np.ndarray:
    """Create RGBA overlay of segmentation on MRI slice."""
    norm = _normalize_slice(mri_2d)
    rgba = np.stack([norm, norm, norm, np.ones_like(norm)], axis=-1)

    for label, color in LABEL_COLORS.items():
        mask = seg_2d == label
        if np.any(mask):
            for c in range(3):
                rgba[mask, c] = (1 - color[3]) * rgba[mask, c] + color[3] * color[c]
    return rgba


def _find_tumor_center(seg_3d: np.ndarray) -> tuple:
    """Find center of mass of tumor region."""
    from scipy import ndimage
    tumor = (seg_3d > 0)
    if np.any(tumor):
        com = ndimage.center_of_mass(tumor)
        return tuple(int(round(c)) for c in com)
    return tuple(s // 2 for s in seg_3d.shape)


def visualize_case_comparison(
    image: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    case_id: str,
    output_dir: str,
    modality_idx: int = 0,
    metrics: Optional[dict] = None,
) -> Path:
    """Visualize ground truth vs prediction for a single case.

    Creates a 2-row x 3-col grid:
        Row 1: Ground Truth overlays (Axial, Coronal, Sagittal)
        Row 2: Prediction overlays (Axial, Coronal, Sagittal)

    Args:
        image: (C, H, W, D) multi-channel MRI volume
        label: (H, W, D) ground truth labels
        prediction: (H, W, D) predicted labels
        case_id: Case identifier
        output_dir: Directory to save the visualization
        modality_idx: Which channel to display (0=t1c by default)
        metrics: Optional dict of metrics to display in title
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mri = image[modality_idx] if image.ndim == 4 else image
    center = _find_tumor_center(label)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor(DARK_BG)

    view_names = ["Axial", "Coronal", "Sagittal"]
    row_names = ["Ground Truth", "Prediction"]

    for row, (seg, row_name) in enumerate(zip([label, prediction], row_names)):
        slices_mri = [
            mri[:, :, center[2]],
            mri[:, center[1], :],
            mri[center[0], :, :],
        ]
        slices_seg = [
            seg[:, :, center[2]],
            seg[:, center[1], :],
            seg[center[0], :, :],
        ]

        for col in range(3):
            ax = axes[row, col]
            ax.set_facecolor(DARK_BG)
            overlay = _overlay_seg_on_mri(slices_mri[col], slices_seg[col])
            ax.imshow(np.rot90(overlay), interpolation="nearest")
            ax.set_title(f"{row_name} - {view_names[col]}", color="white", fontsize=10)
            ax.axis("off")

    # Title with metrics
    title = f"Case: {case_id}"
    if metrics:
        metric_str = " | ".join(f"{k}: {v:.3f}" for k, v in metrics.items() if isinstance(v, float))
        title += f"\n{metric_str}"
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold")

    legend_elements = [
        Patch(facecolor="red", alpha=0.6, label="NCR (1)"),
        Patch(facecolor="green", alpha=0.6, label="Edema (2)"),
        Patch(facecolor=(0, 0.3, 1), alpha=0.6, label="ET (4)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, facecolor=DARK_BG, edgecolor="white",
               labelcolor="white", framealpha=0.8)

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    save_path = output_path / f"{case_id}_comparison.png"
    fig.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    return save_path


def visualize_failure_grid(
    model,
    dataloader,
    failure_case_ids: List[str],
    config: dict,
    output_dir: str,
    device: torch.device,
    max_cases: int = 8,
) -> Path:
    """Generate a grid of failure cases showing GT vs prediction.

    Each row is one failure case with 3 columns:
        MRI | Ground Truth Overlay | Prediction Overlay
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    spatial_size = config["preprocessing"]["spatial_size"]
    sw_batch = config["training"]["sw_batch_size"]
    num_classes = config["data"]["num_classes"]
    post_pred = AsDiscrete(argmax=True)

    cases_shown = 0
    fig_data = []

    with torch.no_grad():
        for batch_data in dataloader:
            case_id = batch_data.get("case_id", ["unknown"])[0]
            if case_id not in failure_case_ids:
                continue

            images = batch_data["image"].to(device)
            labels = batch_data["label"]

            with autocast(enabled=config["training"]["amp"] and device.type == "cuda"):
                outputs = sliding_window_inference(
                    images, spatial_size, sw_batch, inference_wrapper(model), overlap=0.5
                )

            pred = post_pred(outputs[0]).cpu().numpy()
            mri = images[0].cpu().numpy()
            lab = labels[0].cpu().numpy()

            if lab.ndim == 4:
                lab = lab[0]
            if pred.ndim == 4:
                pred = pred[0]

            fig_data.append((case_id, mri, lab, pred))
            cases_shown += 1
            if cases_shown >= max_cases:
                break

    if not fig_data:
        console.print("[yellow]No failure cases found in dataloader[/yellow]")
        return output_path

    n_cases = len(fig_data)
    fig, axes = plt.subplots(n_cases, 3, figsize=(15, 4 * n_cases))
    fig.patch.set_facecolor(DARK_BG)

    if n_cases == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["MRI (T1c)", "Ground Truth", "Prediction"]

    for row, (case_id, mri, lab, pred) in enumerate(fig_data):
        center = _find_tumor_center(lab)
        mri_slice = mri[0, :, :, center[2]]  # T1c axial
        lab_slice = lab[:, :, center[2]]
        pred_slice = pred[:, :, center[2]]

        # MRI only
        ax = axes[row, 0]
        ax.set_facecolor(DARK_BG)
        norm_mri = _normalize_slice(mri_slice)
        ax.imshow(np.rot90(norm_mri), cmap="gray", interpolation="nearest")
        ax.set_title(f"{case_id}\n{col_titles[0]}", color="white", fontsize=9)
        ax.axis("off")

        # GT overlay
        ax = axes[row, 1]
        ax.set_facecolor(DARK_BG)
        ax.imshow(np.rot90(_overlay_seg_on_mri(mri_slice, lab_slice)), interpolation="nearest")
        ax.set_title(col_titles[1], color="white", fontsize=9)
        ax.axis("off")

        # Prediction overlay
        ax = axes[row, 2]
        ax.set_facecolor(DARK_BG)
        ax.imshow(np.rot90(_overlay_seg_on_mri(mri_slice, pred_slice)), interpolation="nearest")
        ax.set_title(col_titles[2], color="white", fontsize=9)
        ax.axis("off")

    legend_elements = [
        Patch(facecolor="red", alpha=0.6, label="NCR (1)"),
        Patch(facecolor="green", alpha=0.6, label="Edema (2)"),
        Patch(facecolor=(0, 0.3, 1), alpha=0.6, label="ET (4)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, facecolor=DARK_BG, edgecolor="white",
               labelcolor="white", framealpha=0.8)

    fig.suptitle("Failure Case Analysis", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    save_path = output_path / "failure_cases_grid.png"
    fig.savefig(save_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved failure grid to {save_path}[/green]")
    return save_path


def plot_metrics_distributions(
    metrics_df: pd.DataFrame,
    config: dict,
    output_dir: str,
) -> Path:
    """Plot Dice score distributions as box plots and histograms."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    regions = list(config["evaluation"]["regions"].keys())
    dice_cols = [f"dice_{r}" for r in regions if f"dice_{r}" in metrics_df.columns]
    class_dice_cols = [c for c in metrics_df.columns if c.startswith("dice_") and c not in dice_cols]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    # 1. Region Dice box plots
    ax = axes[0, 0]
    if dice_cols:
        data = metrics_df[dice_cols].melt(var_name="Region", value_name="Dice")
        data["Region"] = data["Region"].str.replace("dice_", "")
        sns.boxplot(data=data, x="Region", y="Dice", ax=ax, palette="Set2")
        ax.set_title("Dice Score by Tumor Region")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
        ax.legend()

    # 2. Per-class Dice box plots
    ax = axes[0, 1]
    if class_dice_cols:
        data = metrics_df[class_dice_cols].melt(var_name="Class", value_name="Dice")
        data["Class"] = data["Class"].str.replace("dice_", "")
        sns.boxplot(data=data, x="Class", y="Dice", ax=ax, palette="Set3")
        ax.set_title("Dice Score by Class")
        ax.set_ylim(-0.05, 1.05)

    # 3. ET Dice histogram
    ax = axes[1, 0]
    if "dice_ET" in metrics_df.columns:
        ax.hist(metrics_df["dice_ET"].dropna(), bins=20, color="steelblue",
                edgecolor="white", alpha=0.8)
        ax.axvline(x=metrics_df["dice_ET"].mean(), color="red", linestyle="--",
                    label=f"Mean: {metrics_df['dice_ET'].mean():.3f}")
        ax.set_title("ET Dice Distribution")
        ax.set_xlabel("Dice Score")
        ax.set_ylabel("Count")
        ax.legend()

    # 4. Volume scatter: predicted vs true (WT)
    ax = axes[1, 1]
    if "vol_pred_WT" in metrics_df.columns and "vol_true_WT" in metrics_df.columns:
        ax.scatter(metrics_df["vol_true_WT"], metrics_df["vol_pred_WT"],
                   alpha=0.6, c="steelblue", edgecolors="white", s=40)
        max_vol = max(metrics_df["vol_true_WT"].max(), metrics_df["vol_pred_WT"].max())
        ax.plot([0, max_vol], [0, max_vol], "r--", alpha=0.5, label="Perfect")
        ax.set_xlabel("True Volume (voxels)")
        ax.set_ylabel("Predicted Volume (voxels)")
        ax.set_title("Whole Tumor Volume: Predicted vs True")
        ax.legend()

    plt.tight_layout()
    save_path = output_path / "metrics_distributions.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved metrics distributions to {save_path}[/green]")
    return save_path


def plot_training_curves(log_dir: str, output_dir: str) -> Path:
    """Plot training loss and validation Dice curves from TensorBoard logs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try to read from tensorboard events
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training loss
        if "train/loss" in ea.Tags()["scalars"]:
            events = ea.Scalars("train/loss")
            steps = [e.step for e in events]
            values = [e.value for e in events]
            axes[0].plot(steps, values, color="steelblue", alpha=0.8)
            axes[0].set_title("Training Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")

        # Validation Dice
        if "val/mean_dice" in ea.Tags()["scalars"]:
            events = ea.Scalars("val/mean_dice")
            steps = [e.step for e in events]
            values = [e.value for e in events]
            axes[1].plot(steps, values, color="green", alpha=0.8)
            axes[1].set_title("Validation Mean Dice")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Dice Score")

        plt.tight_layout()
        save_path = output_path / "training_curves.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"[green]Saved training curves to {save_path}[/green]")
        return save_path

    except Exception as e:
        console.print(f"[yellow]Could not read TensorBoard logs: {e}[/yellow]")
        return output_path


def plot_model_comparison(
    results: Dict[str, pd.DataFrame],
    output_dir: str,
) -> Path:
    """Compare multiple models side by side.

    Args:
        results: Dict mapping model_name -> metrics DataFrame
        output_dir: Directory to save the plot
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    regions = ["ET", "TC", "WT"]
    model_names = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, region in enumerate(regions):
        col = f"dice_{region}"
        data = []
        for model_name, df in results.items():
            if col in df.columns:
                for val in df[col].dropna():
                    data.append({"Model": model_name, "Dice": val})

        if data:
            plot_df = pd.DataFrame(data)
            sns.boxplot(data=plot_df, x="Model", y="Dice", ax=axes[i], palette="Set2")
            axes[i].set_title(f"{region} Dice Score")
            axes[i].set_ylim(-0.05, 1.05)
            axes[i].axhline(y=0.5, color="red", linestyle="--", alpha=0.3)

    plt.suptitle("Model Comparison - Dice Scores by Region", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = output_path / "model_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved model comparison to {save_path}[/green]")
    return save_path
