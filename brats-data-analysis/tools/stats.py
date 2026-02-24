"""
tools/stats.py — Statistical Analysis

Reads dataset_summary.csv produced by explore.py (or triggers explore if
not found) and produces a comprehensive 5-panel statistical overview:

  Panel 1 — Tumor volume distributions (ET, TC, WT) — histograms
  Panel 2 — Class balance (mean % of voxels per label)
  Panel 3 — Scan-index distribution (patients by number of scans)
  Panel 4 — Tumor volume boxplots grouped by scan index
  Panel 5 — WT vs ET volume scatter (subregion correlation)

Outputs:
  output/stats_overview.png
  output/stats_summary.txt
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from rich.console import Console
from rich.table import Table

console = Console()

BG_COLOR = "#1a1a2e"
ACCENT = "#7b9cff"
COLORS = {
    "ET":  "#4488ff",
    "TC":  "#ff8844",
    "WT":  "#44cc88",
    "NCR": "#ff4444",
    "SNFH": "#44ff88",
}


def _load_or_explore(data_dir: str, output_dir: str) -> pd.DataFrame:
    csv_path = Path(output_dir).expanduser() / "dataset_summary.csv"
    if not csv_path.exists():
        console.print("[yellow]dataset_summary.csv not found — running explore first...[/]")
        from tools.explore import explore_dataset
        explore_dataset(data_dir, output_dir)
    return pd.read_csv(csv_path)


def analyze_stats(data_dir: str, output_dir: str, summary_csv: str | None = None):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_csv:
        df = pd.read_csv(summary_csv)
    else:
        df = _load_or_explore(data_dir, output_dir)

    df_c = df[df["complete"] == True].copy()

    # Derive scan count per patient
    scan_counts = df_c.groupby("patient_id").size().reset_index(name="n_scans")
    df_c = df_c.merge(scan_counts, on="patient_id", how="left")

    fig = plt.figure(figsize=(18, 14), facecolor=BG_COLOR)
    fig.suptitle("BraTS 2024 — Statistical Overview", color="white",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35,
                          left=0.07, right=0.97, top=0.92, bottom=0.07)

    # ── Panel 1: Tumor volume histograms ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0d0d1a")
    for key, col in [("ET", "vol_et_mm3"), ("TC", "vol_tc_mm3"), ("WT", "vol_wt_mm3")]:
        vals = df_c[col].dropna()
        vals = vals[vals > 0]
        ax1.hist(vals, bins=60, alpha=0.65, label=key,
                 color=COLORS[key], edgecolor="none")
    ax1.set_xlabel("Volume (mm³)", color="white", fontsize=9)
    ax1.set_ylabel("Case count", color="white", fontsize=9)
    ax1.set_title("Tumor Region Volume Distributions", color="white", fontsize=10)
    ax1.legend(fontsize=8, facecolor="#2a2a4e", labelcolor="white", framealpha=0.8)
    _style_ax(ax1)

    # ── Panel 2: Class balance bar chart ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0d0d1a")
    total_voxels = (df_c["n_background"].fillna(0) + df_c["n_ncr"].fillna(0) +
                    df_c["n_snfh"].fillna(0) + df_c["n_et"].fillna(0))
    label_keys = ["n_background", "n_ncr", "n_snfh", "n_et"]
    label_names = ["Background", "NCR (1)", "SNFH (2)", "ET (3)"]
    bar_colors = ["#555577", COLORS["NCR"], COLORS["SNFH"], COLORS["ET"]]
    means = []
    for lk in label_keys:
        pct = (df_c[lk].fillna(0) / total_voxels.replace(0, np.nan)) * 100
        means.append(pct.mean())
    bars = ax2.bar(label_names, means, color=bar_colors, edgecolor="none", width=0.6)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1f}%", ha="center", va="bottom", color="white", fontsize=8)
    ax2.set_ylabel("Mean % of total voxels", color="white", fontsize=9)
    ax2.set_title("Class Balance (Mean Voxel %)", color="white", fontsize=10)
    _style_ax(ax2)

    # ── Panel 3: Scan-index distribution ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#0d0d1a")
    scan_dist = scan_counts["n_scans"].value_counts().sort_index()
    ax3.bar(scan_dist.index.astype(str), scan_dist.values,
            color=ACCENT, edgecolor="none", width=0.7)
    for xi, (idx, val) in enumerate(scan_dist.items()):
        ax3.text(xi, val + 1, str(val), ha="center", va="bottom",
                 color="white", fontsize=8)
    ax3.set_xlabel("Number of scans per patient", color="white", fontsize=9)
    ax3.set_ylabel("Patient count", color="white", fontsize=9)
    ax3.set_title("Longitudinal Scan Distribution", color="white", fontsize=10)
    _style_ax(ax3)

    # ── Panel 4: Volume boxplots by scan index ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#0d0d1a")
    scan_indices = sorted(df_c["scan_idx"].unique())
    data_by_scan = [
        df_c[df_c["scan_idx"] == si]["vol_wt_mm3"].dropna().values
        for si in scan_indices
    ]
    data_by_scan = [d[d > 0] for d in data_by_scan]
    bp = ax4.boxplot(
        data_by_scan,
        labels=[str(si) for si in scan_indices],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#aaaacc"),
        capprops=dict(color="#aaaacc"),
        flierprops=dict(marker=".", markersize=2, color="#555577"),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#2a3a5a")
        patch.set_alpha(0.8)
    ax4.set_xlabel("Scan index (100=first scan)", color="white", fontsize=9)
    ax4.set_ylabel("WT Volume (mm³)", color="white", fontsize=9)
    ax4.set_title("WT Volume by Scan Index", color="white", fontsize=10)
    _style_ax(ax4)

    # ── Panel 5: WT vs ET scatter ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#0d0d1a")
    wt = df_c["vol_wt_mm3"].fillna(0)
    et = df_c["vol_et_mm3"].fillna(0)
    mask = (wt > 0) & (et > 0)
    ax5.scatter(wt[mask], et[mask], s=8, alpha=0.4,
                color=ACCENT, edgecolors="none")
    # Trend line
    if mask.sum() > 10:
        m, b = np.polyfit(wt[mask], et[mask], 1)
        xfit = np.linspace(wt[mask].min(), wt[mask].max(), 100)
        ax5.plot(xfit, m * xfit + b, color="#ff8844", linewidth=1.5,
                 label=f"r={np.corrcoef(wt[mask], et[mask])[0,1]:.2f}")
        ax5.legend(fontsize=8, facecolor="#2a2a4e", labelcolor="white", framealpha=0.8)
    ax5.set_xlabel("WT Volume (mm³)", color="white", fontsize=9)
    ax5.set_ylabel("ET Volume (mm³)", color="white", fontsize=9)
    ax5.set_title("Whole Tumor vs Enhancing Tumor", color="white", fontsize=10)
    _style_ax(ax5)

    # ── Panel 6: Per-region volume violin ─────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#0d0d1a")
    violin_data = []
    violin_labels = []
    violin_colors_list = []
    for key, col, color in [
        ("NCR", "vol_ncr_mm3", COLORS["NCR"]),
        ("SNFH", "vol_snfh_mm3", COLORS["SNFH"]),
        ("ET", "vol_et_mm3", COLORS["ET"]),
        ("TC", "vol_tc_mm3", COLORS["TC"]),
        ("WT", "vol_wt_mm3", COLORS["WT"]),
    ]:
        vals = df_c[col].dropna()
        vals = vals[vals > 0].values
        if len(vals) > 1:
            violin_data.append(vals)
            violin_labels.append(key)
            violin_colors_list.append(color)

    if violin_data:
        parts = ax6.violinplot(violin_data, showmedians=True, showextrema=False)
        for i, (pc, color) in enumerate(zip(parts["bodies"], violin_colors_list)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2)
        ax6.set_xticks(range(1, len(violin_labels) + 1))
        ax6.set_xticklabels(violin_labels, color="white", fontsize=8)
        ax6.set_ylabel("Volume (mm³)", color="white", fontsize=9)
    ax6.set_title("Volume Distribution by Region", color="white", fontsize=10)
    _style_ax(ax6)

    out_png = output_dir / "stats_overview.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    console.print(f"[green]Saved:[/] {out_png}")
    console.print(f"[dim]View:[/]  open {out_png}")

    # ── Text summary ──────────────────────────────────────────────────────
    _print_and_save_summary(df_c, output_dir)


def _print_and_save_summary(df_c: pd.DataFrame, output_dir: Path):
    lines = ["BraTS 2024 — Statistical Summary", "=" * 50, ""]

    table = Table(
        title="[bold cyan]BraTS 2024 — Statistical Summary[/]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Mean ± Std", style="white")
    table.add_column("Median", style="white")
    table.add_column("Min / Max", style="dim")

    regions = [
        ("NCR volume (mm³)", "vol_ncr_mm3"),
        ("SNFH volume (mm³)", "vol_snfh_mm3"),
        ("ET volume (mm³)", "vol_et_mm3"),
        ("TC volume (mm³)", "vol_tc_mm3"),
        ("WT volume (mm³)", "vol_wt_mm3"),
    ]

    for label, col in regions:
        vals = df_c[col].dropna()
        mean = vals.mean()
        std = vals.std()
        med = vals.median()
        mn, mx = vals.min(), vals.max()
        table.add_row(
            label,
            f"{mean:,.0f} ± {std:,.0f}",
            f"{med:,.0f}",
            f"{mn:,.0f} / {mx:,.0f}",
        )
        lines.append(f"{label}: mean={mean:.0f} ± {std:.0f}, median={med:.0f}")

    n_total = len(df_c)
    pct_et = 100 * df_c["n_et"].gt(0).sum() / n_total
    pct_snfh = 100 * df_c["n_snfh"].gt(0).sum() / n_total
    pct_no_tumor = 100 * (df_c["has_tumor"] == False).sum() / n_total

    table.add_row("", "", "", "")
    table.add_row("% cases with ET (label 3)", f"{pct_et:.1f}%", "", "")
    table.add_row("% cases with SNFH (label 2)", f"{pct_snfh:.1f}%", "", "")
    table.add_row("% cases with NO tumor", f"{pct_no_tumor:.1f}%", "", "")

    n_patients = df_c["patient_id"].nunique()
    multi_scan_patients = df_c.groupby("patient_id").filter(lambda g: len(g) > 1)["patient_id"].nunique()
    table.add_row("", "", "", "")
    table.add_row("Unique patients", str(n_patients), "", "")
    table.add_row("Patients with >1 scan", str(multi_scan_patients), "", "")

    console.print(table)

    lines += [
        "",
        f"Cases with ET: {pct_et:.1f}%",
        f"Cases with SNFH: {pct_snfh:.1f}%",
        f"Cases with no tumor: {pct_no_tumor:.1f}%",
        f"Unique patients: {n_patients}",
        f"Patients with >1 scan: {multi_scan_patients}",
    ]

    txt_path = output_dir / "stats_summary.txt"
    txt_path.write_text("\n".join(lines))
    console.print(f"[green]Saved:[/] {txt_path}")


def _style_ax(ax):
    ax.tick_params(colors="white", labelsize=8)
    ax.spines["bottom"].set_color("#444466")
    ax.spines["left"].set_color("#444466")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
