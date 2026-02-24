"""
tools/intensity.py — Intensity Distribution Analysis

Samples N cases from the dataset and computes comprehensive intensity
statistics per MRI modality and per tumor region.  Results reveal how
each sequence highlights different tissue types.

Panels:
  1. Overlaid whole-brain intensity histograms for all 4 modalities
  2. Per-modality boxplots: intensity inside each tumor region (4 sub-axes)
  3. Heatmap: mean normalised intensity — modality × region
  4. Violin plots: intensity in ET region across modalities

Outputs:
  output/intensity_analysis.png
  output/intensity_stats.csv
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from rich.console import Console

console = Console()

BG_COLOR = "#1a1a2e"
MODALITIES = ("t1n", "t1c", "t2f", "t2w")
MOD_COLORS = {
    "t1n": "#ff9944",
    "t1c": "#44aaff",
    "t2f": "#44ff88",
    "t2w": "#ff44cc",
}
REGION_LABELS = {
    "background": 0,
    "NCR": 1,
    "SNFH": 2,
    "ET": 3,
    "brain": None,   # all non-zero voxels
}


def _norm(vol: np.ndarray) -> np.ndarray:
    nonzero = vol[vol > 0]
    if len(nonzero) == 0:
        return vol.astype(np.float32)
    lo = np.percentile(nonzero, 1)
    hi = np.percentile(nonzero, 99)
    denom = hi - lo if hi != lo else 1.0
    return (np.clip(vol, lo, hi) - lo) / denom


def _region_stats(vals: np.ndarray, name: str, case_id: str, modality: str) -> dict:
    if len(vals) == 0:
        return {}
    q1, q3 = np.percentile(vals, [25, 75])
    return {
        "case_id": case_id,
        "modality": modality,
        "region": name,
        "n_voxels": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "iqr": float(q3 - q1),
        "p1": float(np.percentile(vals, 1)),
        "p5": float(np.percentile(vals, 5)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "skewness": float(skew(vals)),
        "kurtosis": float(kurtosis(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def analyze_intensity(data_dir: str, output_dir: str, n_sample: int = 50):
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect valid cases (have all 4 modalities + seg)
    cases = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    valid = []
    for c in cases:
        cd = data_dir / c
        if all((cd / f"{c}-{m}.nii.gz").exists() for m in MODALITIES) \
                and (cd / f"{c}-seg.nii.gz").exists():
            valid.append(c)

    n_sample = min(n_sample, len(valid))
    selected = random.sample(valid, n_sample)
    console.print(
        f"Analysing intensity for [bold cyan]{n_sample}[/] sampled cases..."
    )

    all_stats = []
    # For plotting: collect arrays per modality per region
    hist_data: dict[str, list] = {m: [] for m in MODALITIES}
    region_data: dict[str, dict[str, list]] = {
        m: {r: [] for r in REGION_LABELS} for m in MODALITIES
    }

    for case_id in tqdm(selected, desc="Processing cases", unit="case"):
        cd = data_dir / case_id
        try:
            seg = np.asarray(nib.load(str(cd / f"{case_id}-seg.nii.gz")).dataobj,
                             dtype=np.int8)
        except Exception:
            continue

        for m in MODALITIES:
            try:
                arr = np.asarray(
                    nib.load(str(cd / f"{case_id}-{m}.nii.gz")).dataobj,
                    dtype=np.float32,
                )
            except Exception:
                continue

            arr_n = _norm(arr)

            # Collect for histogram
            brain_vals = arr_n[arr > 0]
            hist_data[m].extend(brain_vals[::4].tolist())  # subsample for speed

            for region, lbl in REGION_LABELS.items():
                if lbl is None:
                    mask = arr > 0
                else:
                    mask = seg == lbl

                vals = arr_n[mask]
                if len(vals) > 0:
                    region_data[m][region].extend(vals[::4].tolist())
                    all_stats.append(_region_stats(vals, region, case_id, m))

    # ── Build plots ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor=BG_COLOR)
    fig.suptitle("BraTS 2024 — Intensity Analysis", color="white",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                          left=0.07, right=0.97, top=0.92, bottom=0.07)

    # ── Panel 1: Whole-brain histogram per modality ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0d0d1a")
    for m in MODALITIES:
        vals = np.array(hist_data[m])
        if len(vals) > 0:
            ax1.hist(vals, bins=100, alpha=0.55, label=m.upper(),
                     color=MOD_COLORS[m], density=True, edgecolor="none")
    ax1.set_xlabel("Normalised Intensity", color="white", fontsize=9)
    ax1.set_ylabel("Density", color="white", fontsize=9)
    ax1.set_title("Whole-Brain Intensity Histograms", color="white", fontsize=10)
    ax1.legend(fontsize=8, facecolor="#2a2a4e", labelcolor="white", framealpha=0.8)
    _style_ax(ax1)

    # ── Panel 2: Boxplots per modality × region (2×2 mini-grid) ──────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0d0d1a")
    plot_regions = ["brain", "NCR", "SNFH", "ET"]
    n_mods = len(MODALITIES)
    n_regs = len(plot_regions)
    x = np.arange(n_mods)
    width = 0.18
    offsets = np.linspace(-(n_regs - 1) / 2, (n_regs - 1) / 2, n_regs) * width
    region_colors = ["#aaaacc", "#ff4444", "#44ff88", "#4488ff"]

    for ri, (reg, rc) in enumerate(zip(plot_regions, region_colors)):
        medians = []
        q1s = []
        q3s = []
        for m in MODALITIES:
            vals = np.array(region_data[m][reg])
            if len(vals) > 0:
                medians.append(np.median(vals))
                q1s.append(np.percentile(vals, 25))
                q3s.append(np.percentile(vals, 75))
            else:
                medians.append(0)
                q1s.append(0)
                q3s.append(0)
        xpos = x + offsets[ri]
        ax2.bar(xpos, medians, width=width, color=rc, alpha=0.75,
                label=reg, edgecolor="none")
        ax2.errorbar(xpos, medians,
                     yerr=[np.array(medians) - np.array(q1s),
                           np.array(q3s) - np.array(medians)],
                     fmt="none", ecolor="white", elinewidth=0.8, capsize=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in MODALITIES], color="white", fontsize=8)
    ax2.set_ylabel("Median Normalised Intensity", color="white", fontsize=9)
    ax2.set_title("Intensity by Modality & Region", color="white", fontsize=10)
    ax2.legend(fontsize=7, facecolor="#2a2a4e", labelcolor="white",
               framealpha=0.8, ncol=2)
    _style_ax(ax2)

    # ── Panel 3: Heatmap — mean intensity (modality × region) ─────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#0d0d1a")
    heat_regions = ["background", "NCR", "SNFH", "ET", "brain"]
    heat_matrix = np.zeros((len(MODALITIES), len(heat_regions)))
    for mi, m in enumerate(MODALITIES):
        for ri, reg in enumerate(heat_regions):
            vals = np.array(region_data[m][reg])
            heat_matrix[mi, ri] = np.mean(vals) if len(vals) > 0 else 0.0

    im = ax3.imshow(heat_matrix, cmap="viridis", aspect="auto",
                    vmin=0, vmax=1)
    ax3.set_xticks(range(len(heat_regions)))
    ax3.set_xticklabels(heat_regions, color="white", fontsize=8, rotation=30, ha="right")
    ax3.set_yticks(range(len(MODALITIES)))
    ax3.set_yticklabels([m.upper() for m in MODALITIES], color="white", fontsize=8)
    ax3.set_title("Mean Intensity Heatmap (Modality × Region)", color="white", fontsize=10)
    for mi in range(len(MODALITIES)):
        for ri in range(len(heat_regions)):
            ax3.text(ri, mi, f"{heat_matrix[mi, ri]:.2f}",
                     ha="center", va="center", fontsize=7,
                     color="white" if heat_matrix[mi, ri] < 0.6 else "black")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")
    ax3.tick_params(colors="white")

    # ── Panel 4: Violin — ET region intensity across modalities ───────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#0d0d1a")
    et_vals = [np.array(region_data[m]["ET"]) for m in MODALITIES]
    et_vals_nonempty = [v for v in et_vals if len(v) > 1]
    mods_nonempty = [MODALITIES[i] for i, v in enumerate(et_vals) if len(v) > 1]

    if et_vals_nonempty:
        parts = ax4.violinplot(et_vals_nonempty, showmedians=True, showextrema=True)
        for i, (pc, m) in enumerate(zip(parts["bodies"], mods_nonempty)):
            pc.set_facecolor(MOD_COLORS[m])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2)
        parts["cmaxes"].set_color("#aaaacc")
        parts["cmins"].set_color("#aaaacc")
        parts["cbars"].set_color("#aaaacc")
        ax4.set_xticks(range(1, len(mods_nonempty) + 1))
        ax4.set_xticklabels([m.upper() for m in mods_nonempty], color="white", fontsize=8)
    ax4.set_ylabel("Normalised Intensity", color="white", fontsize=9)
    ax4.set_title("ET Region Intensity per Modality", color="white", fontsize=10)
    _style_ax(ax4)

    out_png = output_dir / "intensity_analysis.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    console.print(f"[green]Saved:[/] {out_png}")
    console.print(f"[dim]View:[/]  open {out_png}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        csv_path = output_dir / "intensity_stats.csv"
        stats_df.to_csv(csv_path, index=False)
        console.print(f"[green]Saved:[/] {csv_path}")

        # Rich summary table
        from rich.table import Table
        tbl = Table(
            title="[bold cyan]Intensity Summary (mean ± std across sampled cases)[/]",
            show_header=True, header_style="bold magenta", border_style="dim",
        )
        tbl.add_column("Modality", style="cyan")
        tbl.add_column("Region", style="white")
        tbl.add_column("Mean", style="white")
        tbl.add_column("Std", style="dim")
        tbl.add_column("Median", style="white")

        agg = stats_df.groupby(["modality", "region"])[["mean", "std", "median"]].mean()
        for (mod, reg), row in agg.iterrows():
            tbl.add_row(mod.upper(), reg,
                        f"{row['mean']:.3f}", f"{row['std']:.3f}", f"{row['median']:.3f}")
        console.print(tbl)


def _style_ax(ax):
    ax.tick_params(colors="white", labelsize=8)
    ax.spines["bottom"].set_color("#444466")
    ax.spines["left"].set_color("#444466")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
