import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage
from rich.console import Console

console = Console()

LABEL_COLORS = {
    1: [1.0, 0.0, 0.0],  # NCR - red
    2: [0.0, 1.0, 0.0],  # Edema - green
    4: [0.0, 0.0, 1.0],  # ET - blue
}

MODALITY_NAMES = ["flair", "t1", "t1ce", "t2"]


def _normalize_slice(s):
    nonzero = s[s > 0]
    if len(nonzero) == 0:
        return np.zeros_like(s)
    p1, p99 = np.percentile(nonzero, [1, 99])
    s = np.clip(s, p1, p99)
    if p99 > p1:
        s = (s - p1) / (p99 - p1)
    return s


def _overlay_seg(mri_slice, seg_slice, alpha=0.5):
    norm = _normalize_slice(mri_slice)
    rgb = np.stack([norm, norm, norm, np.ones_like(norm)], axis=-1)

    for label, color in LABEL_COLORS.items():
        mask = seg_slice == label
        if np.any(mask):
            for c in range(3):
                rgb[mask, c] = (1 - alpha) * rgb[mask, c] + alpha * color[c]

    return rgb


def _find_tumor_center(seg_data):
    tumor_mask = (seg_data == 1) | (seg_data == 2) | (seg_data == 4)
    if np.any(tumor_mask):
        com = ndimage.center_of_mass(tumor_mask)
        return tuple(int(round(c)) for c in com)
    return tuple(s // 2 for s in seg_data.shape)


def visualize_case(data_dir: str, case_id: str, output_dir: str, modality: str = "flair"):
    data_path = Path(data_dir).expanduser()
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    case_dir = data_path / case_id

    if not case_dir.exists():
        console.print(f"[red]Case directory not found: {case_dir}[/red]")
        return

    modalities = {}
    available_mods = []
    for mod in MODALITY_NAMES:
        candidates = list(case_dir.glob(f"*{mod}*.nii.gz"))
        if candidates:
            modalities[mod] = nib.load(str(candidates[0])).get_fdata(dtype=np.float32)
            available_mods.append(mod)

    seg_candidates = list(case_dir.glob("*seg*.nii.gz"))
    if not seg_candidates:
        console.print(f"[red]No segmentation found for {case_id}[/red]")
        return
    seg_data = nib.load(str(seg_candidates[0])).get_fdata(dtype=np.float32)

    if not available_mods:
        console.print(f"[red]No MRI modalities found for {case_id}[/red]")
        return

    center = _find_tumor_center(seg_data)
    num_mods = len(available_mods)

    fig, axes = plt.subplots(num_mods, 3, figsize=(12, 4 * num_mods))
    fig.patch.set_facecolor("#1a1a2e")

    if num_mods == 1:
        axes = axes[np.newaxis, :]

    view_names = ["Axial", "Coronal", "Sagittal"]

    for row, mod in enumerate(available_mods):
        vol = modalities[mod]

        slices_mri = [
            vol[:, :, center[2]],
            vol[:, center[1], :],
            vol[center[0], :, :],
        ]
        slices_seg = [
            seg_data[:, :, center[2]],
            seg_data[:, center[1], :],
            seg_data[center[0], :, :],
        ]

        for col in range(3):
            ax = axes[row, col]
            ax.set_facecolor("#1a1a2e")
            overlay = _overlay_seg(slices_mri[col], slices_seg[col])
            ax.imshow(np.rot90(overlay), interpolation="nearest")
            ax.set_title(f"{mod.upper()} - {view_names[col]}", color="white", fontsize=10)
            ax.axis("off")

    legend_elements = [
        Patch(facecolor="red", label="NCR (1)"),
        Patch(facecolor="green", label="Edema (2)"),
        Patch(facecolor="blue", label="ET (4)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, facecolor="#1a1a2e", edgecolor="white",
               labelcolor="white", framealpha=0.8)

    fig.suptitle(f"Case: {case_id}", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    save_path = output_path / f"{case_id}_visualization.png"
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    console.print(f"[green]Saved visualization to {save_path}[/green]")
    console.print(f"[dim]Open with: open {save_path}[/dim]")
