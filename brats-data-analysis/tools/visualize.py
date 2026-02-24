"""
tools/visualize.py — Single Case Multi-Planar Visualization

Loads all 4 MRI modalities + segmentation mask for a single BraTS 2024
case and produces a 4×3 grid (modality rows × anatomical planes) with
color-coded tumor overlay.

Color scheme:
  NCR  (label 1) = red   (255, 0,   0)
  SNFH (label 2) = green (0,   255, 0)
  ET   (label 3) = blue  (0,   0,   255)
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
from scipy.ndimage import center_of_mass
from rich.console import Console

console = Console()

MODALITIES = ("t1n", "t1c", "t2f", "t2w")
MODALITY_LABELS = {
    "t1n": "T1 Native",
    "t1c": "T1 Contrast",
    "t2f": "T2 FLAIR",
    "t2w": "T2 Weighted",
}
BG_COLOR = "#1a1a2e"
SEG_COLORS = {
    1: (255, 0, 0),    # NCR  — red
    2: (0, 255, 0),    # SNFH — green
    3: (0, 0, 255),    # ET   — blue
}


def _norm(vol: np.ndarray) -> np.ndarray:
    """Percentile-normalize a volume to [0, 1] using non-zero voxels."""
    nonzero = vol[vol > 0]
    if len(nonzero) == 0:
        return vol.astype(np.float32)
    lo = np.percentile(nonzero, 1)
    hi = np.percentile(nonzero, 99)
    clipped = np.clip(vol, lo, hi)
    denom = hi - lo if hi != lo else 1.0
    return (clipped - lo) / denom


def _seg_overlay(slice_2d: np.ndarray, seg_slice: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Build an RGBA image from a grayscale MRI slice + segmentation."""
    gray = np.clip(slice_2d, 0, 1)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgba = np.concatenate([rgb, np.ones((*gray.shape, 1), dtype=np.float32)], axis=-1)

    for lbl, color in SEG_COLORS.items():
        mask = seg_slice == lbl
        if not np.any(mask):
            continue
        for ch, c in enumerate(color):
            rgba[mask, ch] = alpha * (c / 255.0) + (1 - alpha) * gray[mask]
        rgba[mask, 3] = 1.0

    return rgba


def _tumor_center(seg: np.ndarray):
    """Return (z, y, x) center of tumor, or volume center if no tumor."""
    tumor_mask = seg > 0
    if np.any(tumor_mask):
        coords = center_of_mass(tumor_mask)
        return tuple(int(round(c)) for c in coords)
    # Fallback: volume center
    return tuple(s // 2 for s in seg.shape)


def visualize_case(data_dir: str, case_id: str, output_dir: str, modality: str = "t1c"):
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_dir = data_dir / case_id
    if not case_dir.exists():
        console.print(f"[red]Case not found:[/] {case_dir}")
        return

    # Determine which modalities are actually available
    available_mods = []
    for m in MODALITIES:
        p = case_dir / f"{case_id}-{m}.nii.gz"
        if p.exists():
            available_mods.append(m)

    if not available_mods:
        console.print(f"[red]No MRI modalities found for case {case_id}[/]")
        return

    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    if not seg_path.exists():
        console.print(f"[red]Segmentation not found:[/] {seg_path}")
        return

    console.print(f"Loading [bold]{case_id}[/] ({len(available_mods)} modalities)...")

    # Load seg
    seg_img = nib.load(str(seg_path))
    seg = np.asarray(seg_img.dataobj, dtype=np.int8)

    # Find tumor center
    cz, cy, cx = _tumor_center(seg)

    # Load and normalize each modality
    vols = {}
    for m in available_mods:
        img = nib.load(str(case_dir / f"{case_id}-{m}.nii.gz"))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        vols[m] = _norm(arr)

    n_rows = len(available_mods)
    n_cols = 3  # Axial, Coronal, Sagittal
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        facecolor=BG_COLOR,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    plane_titles = ["Axial", "Coronal", "Sagittal"]

    for row_i, m in enumerate(available_mods):
        vol = vols[m]

        # Extract slices at tumor center
        axial_vol = vol[:, :, cz]      if cz < vol.shape[2] else vol[:, :, vol.shape[2]//2]
        axial_seg = seg[:, :, cz]      if cz < seg.shape[2] else seg[:, :, seg.shape[2]//2]

        coronal_vol = vol[:, cy, :]    if cy < vol.shape[1] else vol[:, vol.shape[1]//2, :]
        coronal_seg = seg[:, cy, :]    if cy < seg.shape[1] else seg[:, seg.shape[1]//2, :]

        sagittal_vol = vol[cx, :, :]   if cx < vol.shape[0] else vol[vol.shape[0]//2, :, :]
        sagittal_seg = seg[cx, :, :]   if cx < seg.shape[0] else seg[seg.shape[0]//2, :, :]

        slices = [
            (axial_vol,    axial_seg,    "Axial"),
            (coronal_vol,  coronal_seg,  "Coronal"),
            (sagittal_vol, sagittal_seg, "Sagittal"),
        ]

        for col_i, (sl_vol, sl_seg, plane) in enumerate(slices):
            ax = axes[row_i, col_i]
            ax.set_facecolor(BG_COLOR)

            rgba = _seg_overlay(sl_vol.T, sl_seg.T)
            ax.imshow(rgba, origin="lower", aspect="equal")

            title = f"{MODALITY_LABELS.get(m, m)} — {plane}"
            ax.set_title(title, color="white", fontsize=9, pad=4)
            ax.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=(1.0, 0.0, 0.0), label="NCR  (label 1)"),
        mpatches.Patch(color=(0.0, 1.0, 0.0), label="SNFH (label 2)"),
        mpatches.Patch(color=(0.0, 0.0, 1.0), label="ET   (label 3)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=9,
        facecolor="#2a2a4e",
        labelcolor="white",
        framealpha=0.8,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        f"BraTS 2024 — {case_id}",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = output_dir / f"{case_id}_visualization.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)

    console.print(f"[green]Saved:[/] {out_path}")
    console.print(f"[dim]View:[/]  open {out_path}")
