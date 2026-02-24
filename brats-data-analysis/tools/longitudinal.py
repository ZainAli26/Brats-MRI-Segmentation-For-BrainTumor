"""
tools/longitudinal.py — Longitudinal Patient Analysis

Groups BraTS 2024 cases by patient ID to study tumour progression /
regression across multiple scanning sessions.  BraTS scan indices:
  100 = first scan, 101 = second, 102 = third, …

Panels:
  1. Bar chart  — how many patients have 1 / 2 / 3 / 4 / 5+ scans
  2. Spaghetti  — WT volume trajectory per patient (top N multi-scan)
  3. Histogram  — ΔWT volume (scan[1] − scan[0])  for patients with ≥2 scans

Outputs:
  output/longitudinal_analysis.png
  output/longitudinal_summary.csv
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

BG_COLOR = "#1a1a2e"
ACCENT = "#7b9cff"


def _parse_case(case_id: str):
    parts = case_id.split("-")
    return parts[2], int(parts[3])


def _tumor_volume(seg_path: Path, voxel_vol: float) -> float:
    seg = np.asarray(nib.load(str(seg_path)).dataobj, dtype=np.int8)
    return float(np.sum(seg > 0)) * voxel_vol


def analyze_longitudinal(data_dir: str, output_dir: str):
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    # Group by patient
    patient_cases: dict[str, list] = {}
    for c in cases:
        pid, sidx = _parse_case(c)
        patient_cases.setdefault(pid, []).append((sidx, c))

    for pid in patient_cases:
        patient_cases[pid].sort(key=lambda x: x[0])

    n_patients = len(patient_cases)
    console.print(f"Found [bold cyan]{n_patients}[/] unique patients across [bold cyan]{len(cases)}[/] cases.")

    # Compute WT volume per case (only for multi-scan patients)
    multi_scan = {pid: scans for pid, scans in patient_cases.items() if len(scans) >= 2}
    console.print(f"Patients with ≥2 scans: [bold cyan]{len(multi_scan)}[/]")

    summary_rows = []
    trajectories: dict[str, dict[int, float]] = {}  # pid → {scan_idx: vol_wt}

    for pid, scans in tqdm(
        sorted(multi_scan.items(), key=lambda x: -len(x[1])),
        desc="Computing volumes",
        unit="patient",
    ):
        traj = {}
        for sidx, case_id in scans:
            seg_path = data_dir / case_id / f"{case_id}-seg.nii.gz"
            if not seg_path.exists():
                continue
            try:
                seg_img = nib.load(str(seg_path))
                affine = seg_img.affine
                voxel_vol = float(np.prod(np.abs(np.diag(affine)[:3])))
                vol_wt = _tumor_volume(seg_path, voxel_vol)
                traj[sidx] = vol_wt
            except Exception:
                continue
        if len(traj) >= 2:
            trajectories[pid] = traj

    # Build summary CSV rows
    for pid, traj in trajectories.items():
        sorted_sidx = sorted(traj.keys())
        vols = [traj[s] for s in sorted_sidx]
        first_vol = vols[0]
        last_vol = vols[-1]
        delta = last_vol - first_vol
        delta_pct = (delta / first_vol * 100) if first_vol > 0 else 0.0
        for i, (sidx, vol) in enumerate(zip(sorted_sidx, vols)):
            summary_rows.append({
                "patient_id": pid,
                "n_scans": len(sorted_sidx),
                "scan_idx": sidx,
                "scan_number": i + 1,
                "vol_wt_mm3": round(vol, 2),
                "delta_from_first_mm3": round(vol - first_vol, 2),
                "delta_pct_from_first": round((vol - first_vol) / first_vol * 100
                                               if first_vol > 0 else 0.0, 2),
            })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = output_dir / "longitudinal_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    console.print(f"[green]Saved:[/] {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG_COLOR)

    # Panel 1: Distribution of scan counts
    ax1 = axes[0]
    ax1.set_facecolor("#0d0d1a")
    scan_count_dist = {}
    for pid, scans in patient_cases.items():
        n = len(scans)
        key = str(n) if n < 5 else "5+"
        scan_count_dist[key] = scan_count_dist.get(key, 0) + 1
    sorted_keys = sorted(scan_count_dist.keys(),
                         key=lambda x: int(x.replace("+", "")))
    counts = [scan_count_dist[k] for k in sorted_keys]
    bars = ax1.bar(sorted_keys, counts, color=ACCENT, edgecolor="none", width=0.6)
    for bar, val in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha="center", va="bottom", color="white", fontsize=9)
    ax1.set_xlabel("Number of scans per patient", color="white", fontsize=10)
    ax1.set_ylabel("Patient count", color="white", fontsize=10)
    ax1.set_title("Scan Count Distribution", color="white", fontsize=11)
    _style_ax(ax1)

    # Panel 2: Spaghetti plot (top 30 patients by n_scans)
    ax2 = axes[1]
    ax2.set_facecolor("#0d0d1a")
    top_patients = sorted(trajectories.keys(),
                          key=lambda p: -len(trajectories[p]))[:30]
    cmap = plt.cm.get_cmap("tab20", len(top_patients))
    for i, pid in enumerate(top_patients):
        traj = trajectories[pid]
        x = sorted(traj.keys())
        y = [traj[s] / 1000 for s in x]   # convert to cm³
        ax2.plot(
            [xi - 99 for xi in x],  # normalise: 1 = first scan
            y,
            marker="o", markersize=3, linewidth=1.2,
            color=cmap(i), alpha=0.75,
        )
    ax2.set_xlabel("Scan number (1 = first)", color="white", fontsize=10)
    ax2.set_ylabel("WT Volume (cm³)", color="white", fontsize=10)
    ax2.set_title(f"WT Volume Trajectories (top {len(top_patients)} patients)", color="white", fontsize=11)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _style_ax(ax2)

    # Panel 3: Delta WT (second − first)
    ax3 = axes[2]
    ax3.set_facecolor("#0d0d1a")
    first_scans = df_summary[df_summary["scan_number"] == 1][["patient_id", "vol_wt_mm3"]].rename(
        columns={"vol_wt_mm3": "vol_first"})
    second_scans = df_summary[df_summary["scan_number"] == 2][["patient_id", "vol_wt_mm3"]].rename(
        columns={"vol_wt_mm3": "vol_second"})
    paired = first_scans.merge(second_scans, on="patient_id")
    delta = (paired["vol_second"] - paired["vol_first"]).values

    if len(delta) > 0:
        colors = np.where(delta >= 0, "#ff6644", "#44cc88")
        ax3.hist(delta / 1000, bins=40, color=ACCENT, edgecolor="none", alpha=0.8)
        ax3.axvline(0, color="white", linewidth=1, linestyle="--", alpha=0.6)
        ax3.axvline(np.median(delta / 1000), color="#ff8844", linewidth=1.5,
                    linestyle="-", label=f"Median={np.median(delta)/1000:+.1f} cm³")
        ax3.legend(fontsize=8, facecolor="#2a2a4e", labelcolor="white", framealpha=0.8)

    ax3.set_xlabel("ΔWT Volume scan2−scan1 (cm³)", color="white", fontsize=10)
    ax3.set_ylabel("Patient count", color="white", fontsize=10)
    ax3.set_title("WT Volume Change (scan 2 − scan 1)", color="white", fontsize=11)
    _style_ax(ax3)

    fig.suptitle("BraTS 2024 — Longitudinal Analysis", color="white",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_png = output_dir / "longitudinal_analysis.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    console.print(f"[green]Saved:[/] {out_png}")
    console.print(f"[dim]View:[/]  open {out_png}")

    # Rich summary table
    tbl = Table(
        title="[bold cyan]Longitudinal Summary[/]",
        show_header=True, header_style="bold magenta", border_style="dim",
    )
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", style="white")
    tbl.add_row("Total patients", str(n_patients))
    tbl.add_row("Patients with ≥2 scans", str(len(multi_scan)))
    tbl.add_row("Patients with ≥3 scans",
                str(sum(1 for s in patient_cases.values() if len(s) >= 3)))
    tbl.add_row("Patients with ≥4 scans",
                str(sum(1 for s in patient_cases.values() if len(s) >= 4)))
    if len(delta) > 0:
        tbl.add_row("Median ΔWT scan2−scan1 (cm³)",
                    f"{np.median(delta)/1000:+.2f}")
        tbl.add_row("% patients with WT increase",
                    f"{100*np.mean(delta > 0):.1f}%")
        tbl.add_row("% patients with WT decrease",
                    f"{100*np.mean(delta < 0):.1f}%")
    console.print(tbl)


def _style_ax(ax):
    ax.tick_params(colors="white", labelsize=8)
    ax.spines["bottom"].set_color("#444466")
    ax.spines["left"].set_color("#444466")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
