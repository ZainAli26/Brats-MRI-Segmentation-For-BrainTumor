#!/usr/bin/env python3
"""BraTS Segmentation — Per-Case Visualization with 3D Surface Rendering.

Generates one rich PNG per case showing:
  • All 4 MRI modalities (T1c, T1n, T2f, T2w) at tumor-centred axial slice
  • Ground-truth vs prediction overlays in axial / coronal / sagittal planes
  • 3D surface mesh of each tumour compartment (marching cubes)
  • Per-case Dice bar chart (ET / TC / WT)

Colour scheme
  NCR (class 1) — Red      #FF4444   (necrotic / non-enhancing core)
  ED  (class 2) — Gold     #FFD700   (peritumoral oedema)
  ET  (class 3) — Cyan     #00CFFF   (enhancing tumour)

Usage
-----
# Visualise fold-0 val set (272 cases)  →  results/val_fold0/
python3 visualize_segmentations.py \\
    --fold_dirs runs/nnunet_v2_20260507_021354_fold0 \\
    --config experiments/exp18_nnunet_v2_residual_5fold.yaml \\
    --splits val_fold0

# Visualise held-out test set (≈129 cases)  →  results/test_set/
python3 visualize_segmentations.py \\
    --fold_dirs runs/nnunet_v2_20260507_021354_fold0 \\
    --config experiments/exp18_nnunet_v2_residual_5fold.yaml \\
    --splits test

# Both in one run
python3 visualize_segmentations.py \\
    --fold_dirs runs/nnunet_v2_20260507_021354_fold0 \\
    --config experiments/exp18_nnunet_v2_residual_5fold.yaml \\
    --splits val_fold0 test

# Quick smoke-test (first 5 cases of each split, skip 3-D)
python3 visualize_segmentations.py ... --max_cases 5 --no_3d
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

# Detect whether matplotlib 3-D projection is usable
_HAS_3D = False
try:
    from mpl_toolkits.mplot3d import Axes3D   # registers '3d' projection
    _HAS_3D = True
except Exception:
    pass

import numpy as np
import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from scipy import ndimage
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.data.dataset import build_file_list
from src.data.preprocessing import get_val_transforms
from src.data.splits import create_kfold_splits, create_patient_splits
from src.evaluation.postprocessing import postprocess_prediction
from src.models.factory import create_model
from src.utils import inference_wrapper
from src.utils.experiment import load_config

warnings.filterwarnings("ignore", category=UserWarning)
console = Console()

# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
TEXT_CLR  = "#e6edf3"

# RGBA float tuples used for overlay compositing
SEG_COLORS = {
    1: np.array([1.0,  0.27, 0.27, 0.70]),   # NCR  — red
    2: np.array([1.0,  0.84, 0.0,  0.65]),   # ED   — gold
    3: np.array([0.0,  0.81, 1.0,  0.70]),   # ET   — cyan
}
# Hex strings for 3-D surface and legend
HEX_COLORS = {1: "#FF4545", 2: "#FFD700", 3: "#00CFFF"}
LABEL_NAMES = {1: "NCR (necrotic core)", 2: "ED (oedema)", 3: "ET (enhancing)"}


# ── MRI helpers ───────────────────────────────────────────────────────────────

def _clip_norm(vol: np.ndarray, plo: float = 1, phi: float = 99) -> np.ndarray:
    """Percentile-clip and normalise to [0, 1]."""
    nz = vol[vol > 0]
    if nz.size == 0:
        return np.zeros_like(vol)
    lo, hi = np.percentile(nz, [plo, phi])
    return np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)


def _overlay(mri_2d: np.ndarray, seg_2d: np.ndarray) -> np.ndarray:
    """Composite segmentation colours onto a grey-scale MRI slice → RGBA."""
    norm = _clip_norm(mri_2d)
    rgba = np.stack([norm, norm, norm, np.ones_like(norm)], axis=-1)
    for lbl, col in SEG_COLORS.items():
        m = seg_2d == lbl
        if m.any():
            alpha = col[3]
            rgba[m, :3] = (1 - alpha) * rgba[m, :3] + alpha * col[:3]
    return rgba


def _tumor_center(seg: np.ndarray) -> tuple:
    """Centre of mass of the foreground (falls back to volume centre)."""
    fg = seg > 0
    if fg.any():
        com = ndimage.center_of_mass(fg)
        return tuple(min(int(round(c)), seg.shape[i] - 1) for i, c in enumerate(com))
    return tuple(s // 2 for s in seg.shape)


def _safe_slice(arr, ax, idx):
    """Extract 2-D slice safely along a given axis."""
    idx = min(max(idx, 0), arr.shape[ax] - 1)
    return np.take(arr, idx, axis=ax)


# ── 3-D surface rendering ─────────────────────────────────────────────────────

def _render_3d(ax3d, seg: np.ndarray, title: str, downsample: int = 80):
    """Draw tumour compartment surfaces on a 3-D axis using marching cubes.
    Requires matplotlib mpl_toolkits.mplot3d and scikit-image."""
    try:
        from skimage.measure import marching_cubes
        from skimage.transform import resize as sk_resize
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        ax3d.text(0.5, 0.5, 0.5, "scikit-image\nnot installed",
                  ha="center", va="center", color="white", fontsize=8,
                  transform=ax3d.transAxes)
        return

    h, w, d = seg.shape
    scale = downsample / max(h, w, d)
    new_shape = (max(int(h * scale), 4), max(int(w * scale), 4), max(int(d * scale), 4))

    seg_small = sk_resize(seg.astype(np.float32), new_shape,
                          order=0, anti_aliasing=False, preserve_range=True)
    seg_small = np.round(seg_small).astype(np.int32)

    any_surface = False
    for cls in [2, 1, 3]:   # render ED first (largest), then NCR, ET on top
        mask = (seg_small == cls).astype(np.float32)
        if mask.sum() < 8:
            continue
        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5, allow_degenerate=False)
        except Exception:
            continue
        col = mcolors.to_rgba(HEX_COLORS[cls], alpha=0.75)
        poly = Poly3DCollection(verts[faces], alpha=col[3])
        poly.set_facecolor(col[:3])
        poly.set_edgecolor("none")
        ax3d.add_collection3d(poly)
        any_surface = True

    if any_surface:
        ax3d.set_xlim(0, new_shape[0])
        ax3d.set_ylim(0, new_shape[1])
        ax3d.set_zlim(0, new_shape[2])
        ax3d.view_init(elev=28, azim=225)
    else:
        ax3d.text2D(0.5, 0.5, "no tumour", ha="center", va="center",
                    color="gray", fontsize=9, transform=ax3d.transAxes)

    ax3d.set_facecolor(DARK_BG)
    ax3d.set_axis_off()
    ax3d.set_title(title, color=TEXT_CLR, fontsize=9, pad=2)


def _render_mip(ax, seg: np.ndarray, title: str):
    """Maximum-Intensity Projection fallback when 3-D axes are unavailable.

    Computes colour-coded projections along all 3 axes and tiles them
    into one 2-D panel, giving a clear 3-D impression of tumour shape.
    """
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color=TEXT_CLR, fontsize=8)
    ax.axis("off")

    h, w, d = seg.shape
    # Three projections: axial (XY), coronal (XZ), sagittal (YZ)
    proj_ax  = np.max(seg, axis=2)   # H × W
    proj_cor = np.max(seg, axis=1)   # H × D
    proj_sag = np.max(seg, axis=0)   # W × D

    # Build an RGBA canvas tiling the three projections side-by-side
    total_w = w + d + d + 4   # padding
    total_h = max(h, h, w) + 2
    canvas = np.zeros((total_h, total_w, 4))

    def _seg_to_rgba(seg_2d):
        rgba = np.zeros((*seg_2d.shape, 4))
        for lbl, col in SEG_COLORS.items():
            m = seg_2d == lbl
            rgba[m] = col
        return rgba

    r_ax  = _seg_to_rgba(np.rot90(proj_ax))
    r_cor = _seg_to_rgba(np.rot90(proj_cor))
    r_sag = _seg_to_rgba(np.rot90(proj_sag))

    offsets = [0, r_ax.shape[1] + 2, r_ax.shape[1] + r_cor.shape[1] + 4]
    for rgba, off in zip([r_ax, r_cor, r_sag], offsets):
        rh, rw = rgba.shape[:2]
        canvas[:rh, off:off + rw] = rgba

    ax.imshow(canvas, interpolation="nearest")

    # Small view labels
    for label, x in zip(["Axial", "Coronal", "Sagittal"], offsets):
        ax.text(x + 2, total_h - 2, label, color=TEXT_CLR, fontsize=6, va="bottom")


# ── Dice computation (single case) ───────────────────────────────────────────

def _case_dice(pred: np.ndarray, lab: np.ndarray, regions: dict) -> dict:
    """Compute per-region and per-class Dice for a single case."""
    results = {}

    pred_t = torch.from_numpy(pred).long()
    lab_t  = torch.from_numpy(lab).long()

    for region_name, indices in regions.items():
        pred_r = torch.zeros_like(pred_t, dtype=torch.bool)
        lab_r  = torch.zeros_like(lab_t,  dtype=torch.bool)
        for idx in indices:
            pred_r |= (pred_t == idx)
            lab_r  |= (lab_t  == idx)
        inter = (pred_r & lab_r).sum().float()
        union = pred_r.sum().float() + lab_r.sum().float()
        results[region_name] = (2 * inter / (union + 1e-7)).item()

    # Per-class
    for cls, name in [(1, "NCR"), (2, "ED"), (3, "ET_cls")]:
        inter = ((pred == cls) & (lab == cls)).sum()
        union = (pred == cls).sum() + (lab == cls).sum()
        results[name] = float(2 * inter / max(union, 1))

    return results


# ── per-case figure ───────────────────────────────────────────────────────────

def _make_figure(
    case_id: str,
    image_np: np.ndarray,      # (C, H, W, D)
    label_np: np.ndarray,      # (H, W, D)
    pred_np:  np.ndarray,      # (H, W, D)
    modalities: list,
    regions: dict,
    with_3d: bool,
) -> plt.Figure:
    """Build the full per-case visualisation figure."""

    dice = _case_dice(pred_np, label_np, regions)
    center = _tumor_center(label_np)   # (x, y, z)

    # ── layout ────────────────────────────────────────────────────────────────
    # Columns:
    #   0=mod0  1=mod1  2=mod2  3=mod3  |  4=GT-ax  5=GT-cor  6=GT-sag
    #   7=Pr-ax 8=Pr-cor 9=Pr-sag      |  10=3D-GT  11=3D-Pred  12=metrics
    n_cols  = 13 if with_3d else 10
    fig_w   = 26 if with_3d else 20
    fig_h   = 11 if with_3d else 9

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=DARK_BG)

    # Row heights: main-slices | 3D+metrics
    if with_3d:
        gs_outer = gridspec.GridSpec(
            2, 1, figure=fig, hspace=0.18,
            height_ratios=[2.8, 2.2],
        )
        gs_top = gridspec.GridSpecFromSubplotSpec(
            2, 10, subplot_spec=gs_outer[0], wspace=0.04, hspace=0.06,
        )
        gs_bot = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_outer[1], wspace=0.12,
            width_ratios=[1, 1, 0.7],
        )
    else:
        gs_top = gridspec.GridSpec(2, 10, figure=fig, wspace=0.04, hspace=0.06)

    # ── Row 0: 4 modalities + GT overlays (ax / cor / sag) ───────────────────
    mod_labels = modalities if len(modalities) == 4 else ["T1c", "T1n", "T2f", "T2w"]
    for i, (mod_vol, mod_name) in enumerate(zip(image_np, mod_labels)):
        ax = fig.add_subplot(gs_top[0, i])
        ax.set_facecolor(DARK_BG)
        sl = _safe_slice(mod_vol, 2, center[2])   # axial
        ax.imshow(np.rot90(_clip_norm(sl)), cmap="gray", interpolation="nearest")
        ax.set_title(mod_name.upper(), color=TEXT_CLR, fontsize=8)
        ax.axis("off")

    # GT overlays: ax / cor / sag  (cols 4, 5, 6)
    gt_views = [
        ("Axial",    _safe_slice(image_np[0], 2, center[2]), _safe_slice(label_np, 2, center[2])),
        ("Coronal",  _safe_slice(image_np[0], 1, center[1]), _safe_slice(label_np, 1, center[1])),
        ("Sagittal", _safe_slice(image_np[0], 0, center[0]), _safe_slice(label_np, 0, center[0])),
    ]
    for col_off, (vname, mri_sl, seg_sl) in enumerate(gt_views):
        ax = fig.add_subplot(gs_top[0, 4 + col_off])
        ax.set_facecolor(DARK_BG)
        ax.imshow(np.rot90(_overlay(mri_sl, seg_sl)), interpolation="nearest")
        ax.set_title(f"GT {vname}", color=TEXT_CLR, fontsize=8)
        ax.axis("off")

    # Spacer column 7 → used for divider (leave blank row-0 col-7..9 empty)
    for c in range(7, 10):
        ax = fig.add_subplot(gs_top[0, c])
        ax.set_visible(False)

    # ── Row 1: pred dice text + pred overlays ─────────────────────────────────
    # Region Dice as coloured text (first 4 cols of row-1)
    ax_info = fig.add_subplot(gs_top[1, :4])
    ax_info.set_facecolor(DARK_BG)
    ax_info.axis("off")

    lines = [
        (f"ET  Dice: {dice.get('ET', 0):.3f}",  HEX_COLORS[3]),
        (f"TC  Dice: {dice.get('TC', 0):.3f}",  "#FF9900"),
        (f"WT  Dice: {dice.get('WT', 0):.3f}",  "#AAAAAA"),
        (f"Mean:   {np.mean([dice.get(r, 0) for r in regions]):.3f}", TEXT_CLR),
    ]
    for j, (txt, clr) in enumerate(lines):
        ax_info.text(
            0.05 + (j % 2) * 0.5,
            0.75 - (j // 2) * 0.45,
            txt, color=clr, fontsize=11, fontweight="bold",
            transform=ax_info.transAxes, va="top",
        )

    # Prediction overlays: ax / cor / sag  (cols 4, 5, 6)
    pred_views = [
        ("Axial",    _safe_slice(image_np[0], 2, center[2]), _safe_slice(pred_np, 2, center[2])),
        ("Coronal",  _safe_slice(image_np[0], 1, center[1]), _safe_slice(pred_np, 1, center[1])),
        ("Sagittal", _safe_slice(image_np[0], 0, center[0]), _safe_slice(pred_np, 0, center[0])),
    ]
    for col_off, (vname, mri_sl, seg_sl) in enumerate(pred_views):
        ax = fig.add_subplot(gs_top[1, 4 + col_off])
        ax.set_facecolor(DARK_BG)
        ax.imshow(np.rot90(_overlay(mri_sl, seg_sl)), interpolation="nearest")
        ax.set_title(f"Pred {vname}", color=TEXT_CLR, fontsize=8)
        ax.axis("off")

    # Dice bar chart (cols 7-9, row 1)
    ax_bar = fig.add_subplot(gs_top[1, 7:])
    ax_bar.set_facecolor("#161b22")
    region_names = list(regions.keys())
    region_vals  = [dice.get(r, 0) for r in region_names]
    bar_colors   = ["#00CFFF", "#FF9900", "#AAAAAA"]
    bars = ax_bar.barh(region_names, region_vals, color=bar_colors, edgecolor="none", height=0.5)
    for bar, val in zip(bars, region_vals):
        ax_bar.text(min(val + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left",
                    color=TEXT_CLR, fontsize=9, fontweight="bold")
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_title("Region Dice", color=TEXT_CLR, fontsize=9)
    ax_bar.tick_params(colors=TEXT_CLR, labelsize=8)
    ax_bar.spines[:].set_color("#30363d")
    ax_bar.set_facecolor("#161b22")
    for sp in ax_bar.spines.values():
        sp.set_color("#30363d")
    ax_bar.axvline(x=0.5, color="#FF4545", linestyle="--", lw=0.8, alpha=0.7)

    # ── 3-D bottom row ────────────────────────────────────────────────────────
    if with_3d:
        ax_legend = fig.add_subplot(gs_bot[0, 2])
        ax_legend.set_facecolor(DARK_BG)
        ax_legend.axis("off")

        if _HAS_3D:
            ax_gt3d = fig.add_subplot(gs_bot[0, 0], projection="3d")
            ax_pr3d = fig.add_subplot(gs_bot[0, 1], projection="3d")
            _render_3d(ax_gt3d, label_np, "Ground Truth (3-D)")
            _render_3d(ax_pr3d, pred_np,  "Prediction (3-D)")
        else:
            # Fallback: colour-coded max-intensity projections
            ax_gt_mip = fig.add_subplot(gs_bot[0, 0])
            ax_pr_mip = fig.add_subplot(gs_bot[0, 1])
            _render_mip(ax_gt_mip, label_np, "Ground Truth — MIP (ax / cor / sag)")
            _render_mip(ax_pr_mip, pred_np,  "Prediction  — MIP (ax / cor / sag)")

        # Legend + region explanation
        legend_handles = [
            Patch(color=HEX_COLORS[3], label="ET  — Enhancing Tumour"),
            Patch(color="#FF9900",      label="TC  — Tumour Core  (NCR + ET)"),
            Patch(color="#AAAAAA",      label="WT  — Whole Tumour (NCR + ED + ET)"),
            Patch(color=HEX_COLORS[1], label="NCR — Necrotic Core"),
            Patch(color=HEX_COLORS[2], label="ED  — Peritumoral Oedema"),
        ]
        ax_legend.legend(
            handles=legend_handles, loc="center",
            fontsize=9, facecolor="#161b22", edgecolor="#30363d",
            labelcolor=TEXT_CLR, framealpha=0.9,
        )
        ax_legend.set_title("Tumour Compartments", color=TEXT_CLR, fontsize=9, pad=8)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    mean_dice = np.mean([dice.get(r, 0) for r in regions])
    region_str = "  |  ".join(f"{r}: {dice.get(r,0):.3f}" for r in regions)
    fig.suptitle(
        f"Case: {case_id}\n{region_str}  |  Mean: {mean_dice:.3f}",
        color=TEXT_CLR, fontsize=11, fontweight="bold",
        y=1.01 if with_3d else 1.02,
    )

    # ── Static legend (when no 3-D panel) ─────────────────────────────────────
    if not with_3d:
        legend_handles = [
            Patch(color=HEX_COLORS[1], alpha=0.85, label="NCR"),
            Patch(color=HEX_COLORS[2], alpha=0.85, label="ED"),
            Patch(color=HEX_COLORS[3], alpha=0.85, label="ET"),
        ]
        fig.legend(
            handles=legend_handles, loc="lower center", ncol=3,
            fontsize=9, facecolor=DARK_BG, edgecolor="#30363d",
            labelcolor=TEXT_CLR, framealpha=0.9,
        )

    return fig


# ── inference + save ──────────────────────────────────────────────────────────

def process_cases(
    models: list,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path,
    modalities: list,
    with_3d: bool,
    use_postproc: bool,
    postproc_kwargs: dict,
    max_cases: int = None,
):
    """Run inference on every case and save a visualisation PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_size = config["preprocessing"]["spatial_size"]
    sw_batch     = config["training"]["sw_batch_size"]
    sw_overlap   = config["training"]["sw_overlap"]
    num_classes  = config["data"]["num_classes"]
    regions      = config["evaluation"]["regions"]
    use_amp      = config["training"]["amp"] and device.type == "cuda"

    post_pred_d = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label  = AsDiscrete(to_onehot=num_classes)

    n_done = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Rendering", total=len(dataloader))

        for batch_data in dataloader:
            if max_cases is not None and n_done >= max_cases:
                break

            images  = batch_data["image"].to(device)
            labels  = batch_data["label"].to(device)
            case_id = batch_data.get("case_id", ["unknown"])[0]

            # Ensemble softmax
            with torch.no_grad():
                prob_acc = None
                for m in models:
                    with autocast(enabled=use_amp):
                        logits = sliding_window_inference(
                            images, spatial_size, sw_batch,
                            inference_wrapper(m), overlap=sw_overlap,
                        )
                    prob = torch.softmax(logits, dim=1)
                    prob_acc = prob if prob_acc is None else prob_acc + prob
                ensemble_prob = prob_acc / len(models)

            # Argmax prediction
            outputs_list = decollate_batch(ensemble_prob)
            labels_list  = decollate_batch(labels)
            pred_oh  = post_pred_d(outputs_list[0])   # (C, H, W, D)
            lab_oh   = post_label(labels_list[0])

            pred_np = pred_oh.argmax(dim=0).cpu().numpy().astype(np.int32)
            lab_np  = lab_oh.argmax(dim=0).cpu().numpy().astype(np.int32)

            if use_postproc:
                pred_np = postprocess_prediction(pred_np, **postproc_kwargs)

            image_np = images[0].cpu().numpy()  # (C, H, W, D)

            # Build and save figure
            fig = _make_figure(case_id, image_np, lab_np, pred_np,
                               modalities, regions, with_3d)
            save_path = output_dir / f"{case_id}.png"
            fig.savefig(save_path, dpi=130, facecolor=DARK_BG,
                        bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)

            n_done += 1
            prog.update(task, advance=1)

    console.print(f"[bold green]Saved {n_done} visualisations → {output_dir}[/bold green]")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-case 3-D segmentation visualisations"
    )
    parser.add_argument("--fold_dirs", nargs="+", required=True,
                        help="Fold run directories containing best_model.pth")
    parser.add_argument("--config", required=True,
                        help="Experiment config YAML")
    parser.add_argument("--splits", nargs="+",
                        choices=["test", "val_fold0", "val_fold1",
                                 "val_fold2", "val_fold3", "val_fold4"],
                        default=["val_fold0"],
                        help="Which splits to visualise (can list multiple)")
    parser.add_argument("--output_dir", default="./visualizations",
                        help="Base output directory (default: ./visualizations)")
    parser.add_argument("--no_3d", action="store_true",
                        help="Skip 3-D surface rendering (faster)")
    parser.add_argument("--no_postproc", action="store_true",
                        help="Do not apply post-processing")
    parser.add_argument("--et_min_voxels", type=int, default=250)
    parser.add_argument("--min_component_size", type=int, default=50)
    parser.add_argument("--max_cases", type=int, default=None,
                        help="Process at most N cases per split (for testing)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data.train_dir from config (useful inside Docker)")
    args = parser.parse_args()

    config  = load_config(args.config)
    if args.data_dir:
        config["data"]["train_dir"] = args.data_dir
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with_3d = not args.no_3d

    postproc_kwargs = dict(
        et_min_voxels=args.et_min_voxels,
        min_component_size=args.min_component_size,
        fill_holes=True,
    )

    data_dir   = Path(config["data"]["train_dir"]).expanduser()
    n_folds    = config["data"].get("n_folds", 5)
    modalities = config["data"]["modalities"]
    label_map  = {int(k): int(v) for k, v in config["data"]["label_map"].items()}
    spatial_size = config["preprocessing"]["spatial_size"]
    base_out   = Path(args.output_dir)

    console.print(Panel.fit(
        f"[bold cyan]BraTS Segmentation — Visualisation[/bold cyan]\n"
        f"[dim]Splits: {args.splits} | 3-D surface: {with_3d} | "
        f"Post-proc: {not args.no_postproc} | Device: {device}[/dim]",
        border_style="bright_blue",
    ))

    if not data_dir.exists():
        console.print(f"[red]Data dir not found: {data_dir}[/red]")
        sys.exit(1)

    # Pre-build k-fold splits (used for val_foldX splits)
    kfold_splits = create_kfold_splits(
        str(data_dir), n_folds=n_folds, seed=config["data"]["split_seed"]
    )

    # Load models once (shared across splits)
    models = []
    for fold_dir in [Path(d).expanduser() for d in args.fold_dirs]:
        ckpt_path = fold_dir / "best_model.pth"
        if not ckpt_path.exists():
            console.print(f"[yellow]No checkpoint in {fold_dir}, skipping[/yellow]")
            continue
        m = create_model(config)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device).eval()
        console.print(f"  [green]Loaded {fold_dir.name}[/green] — epoch {ckpt.get('epoch','?')}")
        models.append(m)

    if not models:
        console.print("[red bold]No valid checkpoints found.[/red bold]")
        sys.exit(1)

    val_transform = get_val_transforms(spatial_size, modalities, label_map)

    for split in args.splits:
        # Resolve cases for this split
        if split == "test":
            _, _, eval_cases = create_patient_splits(
                str(data_dir),
                split_ratios=[0.75, 0.15, 0.10],
                seed=config["data"]["split_seed"],
            )
            split_label = "test_set"
        else:
            fold_idx = int(split.split("fold")[-1])
            _, eval_cases = kfold_splits[fold_idx]
            split_label = split   # e.g. "val_fold0"

        if args.max_cases:
            eval_cases = eval_cases[:args.max_cases]

        console.print(
            f"\n[bold]Processing split=[cyan]{split}[/cyan] "
            f"({len(eval_cases)} cases) → [dim]{base_out / split_label}[/dim][/bold]"
        )

        file_list = build_file_list(eval_cases, modalities, include_label=True)
        if not file_list:
            console.print(f"[yellow]No labelled cases found for split {split}[/yellow]")
            continue

        ds = Dataset(file_list, transform=val_transform)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

        process_cases(
            models=models,
            dataloader=loader,
            config=config,
            device=device,
            output_dir=base_out / split_label,
            modalities=modalities,
            with_3d=with_3d,
            use_postproc=not args.no_postproc,
            postproc_kwargs=postproc_kwargs,
            max_cases=args.max_cases,
        )

    console.print(f"\n[bold green]All done. Visualisations in: {base_out}[/bold green]")


if __name__ == "__main__":
    main()
