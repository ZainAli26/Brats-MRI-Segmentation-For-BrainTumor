"""
tools/explore.py — Dataset Inventory

Scans all BraTS 2024 cases, verifies completeness, extracts per-case
metadata (shape, spacing, label voxel counts, tumor volumes) and saves
a dataset_summary.csv for downstream analysis.

BraTS 2024 label schema:
  0 = background
  1 = NCR  (Necrotic Core)
  2 = SNFH (Surrounding Non-enhancing FLAIR Hyperintensity / edema)
  3 = ET   (Enhancing Tumor)

Tumor regions:
  ET  = label 3
  TC  = labels 1 + 3
  WT  = labels 1 + 2 + 3
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

MODALITIES = ("t1n", "t1c", "t2f", "t2w")
SEG_SUFFIX = "-seg.nii.gz"
LABELS = {0: "background", 1: "NCR", 2: "SNFH", 3: "ET"}


def _parse_case_id(case_id: str):
    """Return (patient_id, scan_idx) from 'BraTS-GLI-XXXXX-YYY'."""
    parts = case_id.split("-")
    return parts[2], int(parts[3])


def explore_dataset(data_dir: str, output_dir: str) -> pd.DataFrame:
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    console.print(
        Panel(f"Scanning [bold cyan]{len(cases)}[/] case directories in [dim]{data_dir}[/]",
              title="[bold]BraTS 2024 — Dataset Explore[/]")
    )

    rows = []
    missing_cases = []

    for case_dir in tqdm(cases, desc="Scanning cases", unit="case"):
        case_id = case_dir.name
        patient_id, scan_idx = _parse_case_id(case_id)

        # Check all modality files exist
        mod_paths = {m: case_dir / f"{case_id}-{m}.nii.gz" for m in MODALITIES}
        seg_path = case_dir / f"{case_id}{SEG_SUFFIX}"
        all_files = list(mod_paths.values()) + [seg_path]
        missing = [f.name for f in all_files if not f.exists()]

        if missing:
            missing_cases.append(case_id)
            rows.append({
                "case_id": case_id, "patient_id": patient_id,
                "scan_idx": scan_idx, "complete": False,
                **{k: None for k in [
                    "shape", "spacing_x", "spacing_y", "spacing_z",
                    "n_background", "n_ncr", "n_snfh", "n_et",
                    "vol_ncr_mm3", "vol_snfh_mm3", "vol_et_mm3",
                    "vol_tc_mm3", "vol_wt_mm3", "has_tumor",
                ]},
                "missing_files": ",".join(missing),
            })
            continue

        # Load segmentation only (fast — sparse data)
        seg_img = nib.load(str(seg_path))
        seg = np.asarray(seg_img.dataobj, dtype=np.int8)
        shape = seg.shape
        affine = seg_img.affine
        # Voxel spacings from affine diagonal
        spacings = np.abs(np.diag(affine)[:3])

        voxel_vol = float(np.prod(spacings))  # mm³ per voxel

        counts = {lbl: int(np.sum(seg == lbl)) for lbl in LABELS}

        vol_ncr = counts[1] * voxel_vol
        vol_snfh = counts[2] * voxel_vol
        vol_et = counts[3] * voxel_vol
        vol_tc = (counts[1] + counts[3]) * voxel_vol
        vol_wt = (counts[1] + counts[2] + counts[3]) * voxel_vol
        has_tumor = (counts[1] + counts[2] + counts[3]) > 0

        rows.append({
            "case_id": case_id,
            "patient_id": patient_id,
            "scan_idx": scan_idx,
            "complete": True,
            "shape": f"{shape[0]}x{shape[1]}x{shape[2]}",
            "spacing_x": round(spacings[0], 4),
            "spacing_y": round(spacings[1], 4),
            "spacing_z": round(spacings[2], 4),
            "n_background": counts[0],
            "n_ncr": counts[1],
            "n_snfh": counts[2],
            "n_et": counts[3],
            "vol_ncr_mm3": round(vol_ncr, 2),
            "vol_snfh_mm3": round(vol_snfh, 2),
            "vol_et_mm3": round(vol_et, 2),
            "vol_tc_mm3": round(vol_tc, 2),
            "vol_wt_mm3": round(vol_wt, 2),
            "has_tumor": has_tumor,
            "missing_files": "",
        })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "dataset_summary.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"\n[green]Saved:[/] {csv_path}")

    _print_summary_table(df, len(cases), missing_cases)
    return df


def _print_summary_table(df: pd.DataFrame, total: int, missing_cases: list):
    complete_df = df[df["complete"] == True]
    n_complete = len(complete_df)

    # Most common shape
    shape_counts = complete_df["shape"].value_counts()
    most_common_shape = shape_counts.index[0] if len(shape_counts) > 0 else "N/A"

    # Most common spacing (round to 1 decimal for grouping)
    def fmt_spacing(row):
        return f"{row.spacing_x:.1f}x{row.spacing_y:.1f}x{row.spacing_z:.1f}"
    spacing_series = complete_df.apply(fmt_spacing, axis=1)
    most_common_spacing = spacing_series.value_counts().index[0] if len(spacing_series) > 0 else "N/A"

    n_with_et = int(complete_df["n_et"].gt(0).sum())
    n_with_snfh = int(complete_df["n_snfh"].gt(0).sum())
    n_with_tumor = int(complete_df["has_tumor"].sum())
    n_patients = complete_df["patient_id"].nunique()

    table = Table(
        title="[bold cyan]BraTS 2024 Dataset Summary[/]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Total cases", str(total))
    table.add_row("Complete cases", f"[green]{n_complete}[/]")
    table.add_row("Incomplete / missing files", f"[red]{len(missing_cases)}[/]")
    table.add_row("Unique patients", str(n_patients))
    table.add_row("Most common image shape", most_common_shape)
    table.add_row("Most common voxel spacing (mm)", most_common_spacing)
    table.add_row("Cases with ET  (label 3)", f"{n_with_et}  ({100*n_with_et/n_complete:.1f}%)")
    table.add_row("Cases with SNFH (label 2)", f"{n_with_snfh}  ({100*n_with_snfh/n_complete:.1f}%)")
    table.add_row("Cases with any tumor", f"{n_with_tumor}  ({100*n_with_tumor/n_complete:.1f}%)")
    table.add_row(
        "Median WT volume (mm³)",
        f"{complete_df['vol_wt_mm3'].median():,.0f}"
    )
    table.add_row(
        "Median ET volume (mm³)",
        f"{complete_df['vol_et_mm3'].median():,.0f}"
    )

    console.print(table)

    if missing_cases:
        console.print(f"\n[yellow]Incomplete cases:[/] {', '.join(missing_cases[:10])}"
                      + (" ..." if len(missing_cases) > 10 else ""))
