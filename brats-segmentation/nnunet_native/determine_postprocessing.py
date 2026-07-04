#!/usr/bin/env python3
"""Determine nnU-Net-style post-processing from a folder of predicted segmentations.

Mirrors `nnUNetv2_determine_postprocessing`: it reads predicted label NIfTIs (e.g. the
out-of-fold VALIDATION predictions), pairs them with ground truth, greedily selects
"remove all but largest connected component" operations that improve mean region Dice,
and writes `postprocessing.json`. That file is then applied to the held-out TEST
predictions (via evaluate_nnunet.py --postprocessing_json) so determination never sees
test data — no leakage.

Usage:
    # 1) determine from out-of-fold validation predictions
    python nnunet_native/determine_postprocessing.py \
        --pred_dir nnunet_data/.../fold_0/validation \
        --data_dir ../Brats2024/training_data1_v2 \
        --out_json runs/exp19.../postprocessing.json

    # 2) apply during test evaluation
    python nnunet_native/evaluate_nnunet.py --pred_dir .../test_predictions \
        --data_dir ../Brats2024/training_data1_v2 \
        --postprocessing_json runs/exp19.../postprocessing.json
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.nnunet_postprocessing import determine_postprocessing

console = Console()

# BraTS 2024 post-treatment defaults (override via --num_classes for the 2023 scheme).
DEFAULT_REGIONS = {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3], "RC": [4]}


def _load_pairs(pred_dir: Path, data_dir: Path):
    preds, gts, ids = [], [], []
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        console.print(f"[red]No prediction NIfTIs in {pred_dir}[/red]")
        return preds, gts, ids
    for pf in tqdm(pred_files, desc="Loading"):
        case_id = pf.name[: -len(".nii.gz")]
        case_dir = data_dir / case_id
        seg = list(case_dir.glob("*-seg.nii.gz")) if case_dir.exists() else []
        if not seg:
            console.print(f"[yellow]No GT for {case_id}, skipping[/yellow]")
            continue
        preds.append(nib.load(str(pf)).get_fdata().astype(np.uint8))
        gts.append(nib.load(str(seg[0])).get_fdata().astype(np.uint8))
        ids.append(case_id)
    return preds, gts, ids


def main():
    ap = argparse.ArgumentParser(description="Determine nnU-Net-style post-processing")
    ap.add_argument("--pred_dir", required=True, help="Folder of predicted label NIfTIs (use OOF validation)")
    ap.add_argument("--data_dir", default="../Brats2024/training_data1_v2", help="GT data dir")
    ap.add_argument("--out_json", required=True, help="Where to write postprocessing.json")
    ap.add_argument("--num_classes", type=int, default=5, help="5=BraTS2024, 4=BraTS2023")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    foreground = list(range(1, args.num_classes))
    regions = {k: v for k, v in DEFAULT_REGIONS.items()
               if all(l < args.num_classes for l in v)}

    preds, gts, ids = _load_pairs(pred_dir, data_dir)
    if not preds:
        sys.exit(1)
    console.print(f"\n[bold]Determining post-processing on {len(preds)} validation cases...[/bold]")

    ops, report = determine_postprocessing(preds, gts, foreground, regions)

    table = Table(title="Post-processing determination", style="bold magenta")
    table.add_column("Candidate"); table.add_column("Mean Dice", justify="right"); table.add_column("Decision")
    for entry in report["log"]:
        table.add_row(entry["op"], f"{entry['dice']:.4f}",
                      "[green]ACCEPT[/green]" if entry["accepted"] else "[dim]reject[/dim]")
    console.print(table)
    console.print(f"baseline={report['baseline_dice']:.4f}  ->  final={report['final_dice']:.4f}  "
                  f"(gain {report['gain']:+.4f})")

    out = Path(args.out_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"operations": ops, "regions": regions, "num_classes": args.num_classes,
                   "report": report, "n_val_cases": len(preds)}, f, indent=2)
    console.print(f"[green]Saved -> {out}[/green]")
    if not ops:
        console.print("[dim]No operation improved validation Dice — test eval will be unchanged.[/dim]")


if __name__ == "__main__":
    main()
