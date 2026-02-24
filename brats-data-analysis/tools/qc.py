"""
tools/qc.py — Quality Control

Performs a systematic quality-control sweep across all cases, checking:

  • File completeness     — all 5 modalities present
  • Image shape           — flag non-standard shapes
  • Voxel spacing         — flag anisotropic or extreme values
  • Empty segmentation    — no foreground labels at all
  • Near-zero variance    — a modality that is essentially constant
  • Intensity outliers    — gross outliers in file size or voxel range

Outputs:
  output/qc_report.csv      — per-case quality flags
  output/qc_summary.png     — summary visualisation
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

BG_COLOR = "#1a1a2e"
MODALITIES = ("t1n", "t1c", "t2f", "t2w")

# Expected shape and spacing ranges
EXPECTED_SHAPE = (240, 240, 155)
SPACING_MIN, SPACING_MAX = 0.5, 2.5      # mm, per axis
SPACING_ISO_TOL = 0.2                    # tolerance for anisotropy flag


def run_qc(data_dir: str, output_dir: str):
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    console.print(f"Running QC on [bold cyan]{len(cases)}[/] cases...")

    rows = []

    for case_id in tqdm(cases, desc="QC sweep", unit="case"):
        cd = data_dir / case_id
        row = {
            "case_id": case_id,
            "flag_missing_files": False,
            "flag_shape_mismatch": False,
            "flag_anisotropic_spacing": False,
            "flag_extreme_spacing": False,
            "flag_empty_seg": False,
            "flag_low_variance": False,
            "flag_intensity_outlier": False,
            "shape": None,
            "spacing": None,
            "n_tumor_voxels": None,
            "notes": [],
        }

        # ── 1. File completeness ───────────────────────────────────────────
        seg_path = cd / f"{case_id}-seg.nii.gz"
        mod_paths = {m: cd / f"{case_id}-{m}.nii.gz" for m in MODALITIES}
        missing = [f.name for f in list(mod_paths.values()) + [seg_path] if not f.exists()]
        if missing:
            row["flag_missing_files"] = True
            row["notes"].append(f"Missing: {','.join(missing)}")
            rows.append(_finalise(row))
            continue

        # ── 2. Shape and spacing from seg ──────────────────────────────────
        try:
            seg_img = nib.load(str(seg_path))
            seg = np.asarray(seg_img.dataobj, dtype=np.int8)
            affine = seg_img.affine
            spacings = np.abs(np.diag(affine)[:3])
        except Exception as e:
            row["notes"].append(f"seg load error: {e}")
            rows.append(_finalise(row))
            continue

        shape = seg.shape
        row["shape"] = f"{shape[0]}x{shape[1]}x{shape[2]}"
        row["spacing"] = f"{spacings[0]:.3f}x{spacings[1]:.3f}x{spacings[2]:.3f}"

        if shape != EXPECTED_SHAPE:
            row["flag_shape_mismatch"] = True
            row["notes"].append(f"Shape {shape} != expected {EXPECTED_SHAPE}")

        # Spacing checks
        if any(s < SPACING_MIN or s > SPACING_MAX for s in spacings):
            row["flag_extreme_spacing"] = True
            row["notes"].append(f"Extreme spacing: {spacings.tolist()}")

        sp_range = spacings.max() - spacings.min()
        if sp_range > SPACING_ISO_TOL:
            row["flag_anisotropic_spacing"] = True
            row["notes"].append(f"Anisotropic spacing range: {sp_range:.3f} mm")

        # ── 3. Empty segmentation ──────────────────────────────────────────
        n_tumor = int(np.sum(seg > 0))
        row["n_tumor_voxels"] = n_tumor
        if n_tumor == 0:
            row["flag_empty_seg"] = True
            row["notes"].append("Empty segmentation (no tumor labels)")

        # ── 4. Near-zero variance in any modality ─────────────────────────
        for m in MODALITIES:
            try:
                arr = np.asarray(
                    nib.load(str(mod_paths[m])).dataobj, dtype=np.float32
                )
                nonzero = arr[arr > 0]
                if len(nonzero) == 0:
                    row["flag_low_variance"] = True
                    row["notes"].append(f"{m}: all zeros")
                elif np.std(nonzero) < 1e-3:
                    row["flag_low_variance"] = True
                    row["notes"].append(f"{m}: near-zero variance ({np.std(nonzero):.2e})")
            except Exception as e:
                row["notes"].append(f"{m} load error: {e}")

        # ── 5. Intensity outlier (max value > 99th pct × 10) ──────────────
        for m in MODALITIES:
            try:
                arr = np.asarray(
                    nib.load(str(mod_paths[m])).dataobj, dtype=np.float32
                )
                nonzero = arr[arr > 0]
                if len(nonzero) == 0:
                    continue
                p99 = np.percentile(nonzero, 99)
                max_val = nonzero.max()
                if p99 > 0 and max_val > p99 * 10:
                    row["flag_intensity_outlier"] = True
                    row["notes"].append(
                        f"{m}: max={max_val:.0f} >> p99={p99:.0f} (ratio {max_val/p99:.1f}x)"
                    )
            except Exception:
                pass

        rows.append(_finalise(row))

    df = pd.DataFrame(rows)
    csv_path = output_dir / "qc_report.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]Saved:[/] {csv_path}")

    _print_qc_table(df)
    _plot_qc_summary(df, output_dir)


def _finalise(row: dict) -> dict:
    row["notes"] = " | ".join(row["notes"]) if row["notes"] else ""
    row["n_flags"] = sum(1 for k, v in row.items()
                         if k.startswith("flag_") and v is True)
    return row


def _print_qc_table(df: pd.DataFrame):
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    tbl = Table(
        title="[bold cyan]QC Report Summary[/]",
        show_header=True, header_style="bold magenta", border_style="dim",
    )
    tbl.add_column("Flag", style="cyan")
    tbl.add_column("Affected cases", style="white")
    tbl.add_column("Percentage", style="dim")

    n_total = len(df)
    for fc in flag_cols:
        n_flagged = int(df[fc].sum())
        label = fc.replace("flag_", "").replace("_", " ").title()
        color = "[red]" if n_flagged > 0 else "[green]"
        tbl.add_row(
            label,
            f"{color}{n_flagged}[/]",
            f"{100*n_flagged/n_total:.1f}%",
        )

    tbl.add_section()
    n_clean = int((df["n_flags"] == 0).sum())
    tbl.add_row("[bold]Clean cases (0 flags)[/]",
                f"[green]{n_clean}[/]", f"{100*n_clean/n_total:.1f}%")
    tbl.add_row("[bold]Cases with ≥1 flag[/]",
                f"[yellow]{n_total - n_clean}[/]",
                f"{100*(n_total-n_clean)/n_total:.1f}%")
    console.print(tbl)


def _plot_qc_summary(df: pd.DataFrame, output_dir: Path):
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    labels = [c.replace("flag_", "").replace("_", " ").title() for c in flag_cols]
    counts = [int(df[fc].sum()) for fc in flag_cols]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG_COLOR)

    # Panel 1: Flag counts bar chart
    ax1 = axes[0]
    ax1.set_facecolor("#0d0d1a")
    colors = ["#ff4444" if c > 0 else "#44cc88" for c in counts]
    bars = ax1.barh(labels[::-1], counts[::-1], color=colors[::-1],
                    edgecolor="none", height=0.6)
    for bar, val in zip(bars, counts[::-1]):
        if val > 0:
            ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", ha="left", color="white", fontsize=8)
    ax1.set_xlabel("Flagged cases", color="white", fontsize=9)
    ax1.set_title("QC Flag Counts", color="white", fontsize=11)
    _style_ax(ax1)

    # Panel 2: Shape distribution
    ax2 = axes[1]
    ax2.set_facecolor("#0d0d1a")
    shapes = df["shape"].dropna().value_counts()
    top_shapes = shapes.head(10)
    ax2.barh(top_shapes.index[::-1], top_shapes.values[::-1],
             color="#7b9cff", edgecolor="none", height=0.6)
    ax2.set_xlabel("Case count", color="white", fontsize=9)
    ax2.set_title("Image Shape Distribution", color="white", fontsize=11)
    _style_ax(ax2)

    # Panel 3: n_tumor_voxels distribution
    ax3 = axes[2]
    ax3.set_facecolor("#0d0d1a")
    tumor_voxels = df["n_tumor_voxels"].dropna()
    tumor_voxels = tumor_voxels[tumor_voxels > 0]
    if len(tumor_voxels) > 0:
        ax3.hist(tumor_voxels / 1000, bins=60, color="#44aaff",
                 edgecolor="none", alpha=0.8)
        ax3.axvline(tumor_voxels.median() / 1000, color="#ff8844",
                    linewidth=1.5, linestyle="--",
                    label=f"Median={tumor_voxels.median()/1000:.1f}k")
        ax3.legend(fontsize=8, facecolor="#2a2a4e", labelcolor="white", framealpha=0.8)
    ax3.set_xlabel("Tumor voxels (thousands)", color="white", fontsize=9)
    ax3.set_ylabel("Case count", color="white", fontsize=9)
    ax3.set_title("Tumor Voxel Count Distribution", color="white", fontsize=11)
    _style_ax(ax3)

    fig.suptitle("BraTS 2024 — Quality Control Summary", color="white",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_png = output_dir / "qc_summary.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    console.print(f"[green]Saved:[/] {out_png}")
    console.print(f"[dim]View:[/]  open {out_png}")


def _style_ax(ax):
    ax.tick_params(colors="white", labelsize=8)
    ax.spines["bottom"].set_color("#444466")
    ax.spines["left"].set_color("#444466")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
