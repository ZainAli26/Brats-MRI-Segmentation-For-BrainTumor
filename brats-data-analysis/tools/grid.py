"""
tools/grid.py — Multi-Case Overview Grid

Randomly samples N cases from the dataset and renders each as a single
axial slice (at tumor center) with segmentation overlay.  Layout is 3
columns × ceil(N/3) rows.  Useful for a quick sanity-check across many
cases.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
from scipy.ndimage import center_of_mass
from tqdm import tqdm
from rich.console import Console

console = Console()

BG_COLOR = "#1a1a2e"
SEG_COLORS = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}
MODALITIES = ("t1n", "t1c", "t2f", "t2w")


def _norm(vol: np.ndarray) -> np.ndarray:
    nonzero = vol[vol > 0]
    if len(nonzero) == 0:
        return vol.astype(np.float32)
    lo = np.percentile(nonzero, 1)
    hi = np.percentile(nonzero, 99)
    denom = hi - lo if hi != lo else 1.0
    return (np.clip(vol, lo, hi) - lo) / denom


def _seg_overlay(gray: np.ndarray, seg: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    gray = np.clip(gray, 0, 1)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgba = np.concatenate([rgb, np.ones((*gray.shape, 1), dtype=np.float32)], axis=-1)
    for lbl, color in SEG_COLORS.items():
        mask = seg == lbl
        if not np.any(mask):
            continue
        for ch, c in enumerate(color):
            rgba[mask, ch] = alpha * (c / 255.0) + (1 - alpha) * gray[mask]
    return rgba


def _best_axial_slice(seg: np.ndarray, vol: np.ndarray) -> int:
    """Return axial slice index at tumor center (or volume center)."""
    tumor_mask = seg > 0
    if np.any(tumor_mask):
        cz = int(round(center_of_mass(tumor_mask)[2]))
        return min(max(cz, 0), vol.shape[2] - 1)
    return vol.shape[2] // 2


def visualize_grid(
    data_dir: str,
    output_dir: str,
    n: int = 6,
    modality: str = "t1c",
):
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if modality not in MODALITIES:
        console.print(f"[red]Unknown modality '{modality}'. Choose from: {MODALITIES}[/]")
        return

    # Collect all valid cases
    cases = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    valid = [
        c for c in cases
        if (data_dir / c / f"{c}-{modality}.nii.gz").exists()
        and (data_dir / c / f"{c}-seg.nii.gz").exists()
    ]

    if len(valid) == 0:
        console.print("[red]No valid cases found.[/]")
        return

    n = min(n, len(valid))
    selected = random.sample(valid, n)

    n_cols = 3
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 4.2),
        facecolor=BG_COLOR,
    )
    # Normalise axes to always be 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, case_id in enumerate(
        tqdm(selected, desc=f"Rendering {n} cases", unit="case")
    ):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        ax.set_facecolor(BG_COLOR)

        try:
            seg_img = nib.load(str(data_dir / case_id / f"{case_id}-seg.nii.gz"))
            seg = np.asarray(seg_img.dataobj, dtype=np.int8)

            vol_img = nib.load(str(data_dir / case_id / f"{case_id}-{modality}.nii.gz"))
            vol = np.asarray(vol_img.dataobj, dtype=np.float32)
            vol = _norm(vol)

            z = _best_axial_slice(seg, vol)
            sl_vol = vol[:, :, z]
            sl_seg = seg[:, :, z]

            rgba = _seg_overlay(sl_vol.T, sl_seg.T)
            ax.imshow(rgba, origin="lower", aspect="equal")
            ax.set_title(case_id, color="white", fontsize=7, pad=3)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{case_id[:15]}", color="red",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)

        ax.axis("off")

    # Hide empty cells
    total_cells = n_rows * n_cols
    for idx in range(n, total_cells):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color=(1.0, 0.0, 0.0), label="NCR  (1)"),
        mpatches.Patch(color=(0.0, 1.0, 0.0), label="SNFH (2)"),
        mpatches.Patch(color=(0.0, 0.0, 1.0), label="ET   (3)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=8,
        facecolor="#2a2a4e",
        labelcolor="white",
        framealpha=0.8,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        f"BraTS 2024 — {n} Cases — {modality.upper()} Axial",
        color="white", fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    out_path = output_dir / f"grid_{n}cases_{modality}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)

    console.print(f"[green]Saved:[/] {out_path}")
    console.print(f"[dim]View:[/]  open {out_path}")
