import os
from pathlib import Path
from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

MODALITIES = ["flair", "t1", "t1ce", "t2", "seg"]


def explore_dataset(data_dir: str, output_dir: str):
    data_path = Path(data_dir).expanduser()
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not case_dirs:
        console.print("[red]No case directories found.[/red]")
        return

    records = []
    complete_cases = []
    missing_cases = []
    shapes = []
    spacings = []

    for case_dir in tqdm(case_dirs, desc="Scanning cases"):
        case_id = case_dir.name
        found = {}
        for mod in MODALITIES:
            candidates = list(case_dir.glob(f"*{mod}*.nii.gz"))
            if candidates:
                found[mod] = candidates[0]

        missing_mods = [m for m in MODALITIES if m not in found]

        if missing_mods:
            missing_cases.append(case_id)
            records.append({
                "case_id": case_id,
                "complete": False,
                "missing": ", ".join(missing_mods),
                "shape": None,
                "voxel_spacing": None,
                "et_voxels": 0,
                "ed_voxels": 0,
            })
            continue

        complete_cases.append(case_id)

        seg_img = nib.load(str(found["seg"]))
        seg_data = seg_img.get_fdata(dtype=np.float32)
        shape = seg_img.shape
        spacing = tuple(np.round(seg_img.header.get_zooms()[:3], 2))
        shapes.append(shape)
        spacings.append(spacing)

        et_count = int(np.sum(seg_data == 4))
        ed_count = int(np.sum(seg_data == 2))

        records.append({
            "case_id": case_id,
            "complete": True,
            "missing": "",
            "shape": str(shape),
            "voxel_spacing": str(spacing),
            "et_voxels": et_count,
            "ed_voxels": ed_count,
        })

    df = pd.DataFrame(records)
    csv_path = output_path / "dataset_summary.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]Saved summary to {csv_path}[/green]")

    most_common_shape = Counter(shapes).most_common(1)[0][0] if shapes else "N/A"
    most_common_spacing = Counter(spacings).most_common(1)[0][0] if spacings else "N/A"
    cases_with_et = sum(1 for r in records if r["et_voxels"] > 0)
    cases_with_ed = sum(1 for r in records if r["ed_voxels"] > 0)

    table = Table(title="BraTS Dataset Summary", style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total cases", str(len(case_dirs)))
    table.add_row("Complete cases", str(len(complete_cases)))
    table.add_row("Missing cases", str(len(missing_cases)))
    table.add_row("Most common shape", str(most_common_shape))
    table.add_row("Voxel spacing", str(most_common_spacing))
    table.add_row("Cases with ET (label 4)", str(cases_with_et))
    table.add_row("Cases with Edema (label 2)", str(cases_with_ed))
    console.print(table)
