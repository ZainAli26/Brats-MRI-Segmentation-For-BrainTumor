#!/usr/bin/env python3
"""Evaluate native nnU-Net v2 5-fold CV (exp19) using our shared metrics.

nnU-Net writes each fold's validation predictions to
    <results>/<TRAINER__PLANS__CONFIG>/fold_<k>/validation/<case>.nii.gz
Because the 5 folds partition the dataset at patient level, the union of those
validation predictions is a full-dataset, OUT-OF-FOLD prediction set — exactly
the quantity exp18's `evaluate_kfold.py` reports. We load every fold's validation
preds, pair them with ground truth, and compute the same per-case / per-class /
per-region Dice and HD95 used everywhere else, then write a run dir that
`analyze_failures.py --compare` can read alongside exp18.

Usage:
    python nnunet_native/evaluate_nnunet_kfold.py \
        --results_dir nnunet_data/nnUNet_results/Dataset102_BraTS2024ResEnc/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres \
        --data_dir ../Brats2024/training_data1_v2 \
        --output_dir runs/exp19_nnunet_native_resenc_eval \
        --n_folds 5
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.failure_analysis import identify_failure_cases, print_failure_summary
from src.evaluation.visualization import plot_metrics_distributions

console = Console()

# Identity remap — BraTS 2024 post-treatment labels are already contiguous 0..4
# (1=NETC, 2=SNFH, 3=ET, 4=RC).
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
REGIONS = {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3], "RC": [4]}
CLASS_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}


def _dice(pred_mask, true_mask):
    intersection = np.sum(pred_mask & true_mask)
    total = np.sum(pred_mask) + np.sum(true_mask)
    if total == 0:
        return 1.0 if np.sum(true_mask) == 0 else 0.0
    return 2.0 * intersection / total


def _hausdorff95(pred_mask, true_mask):
    """95th-percentile symmetric Hausdorff distance (voxels)."""
    pred_any, true_any = np.any(pred_mask), np.any(true_mask)
    if not pred_any and not true_any:
        return 0.0
    if not pred_any or not true_any:
        return np.nan

    pred_surface = pred_mask ^ ndimage.binary_erosion(pred_mask)
    true_surface = true_mask ^ ndimage.binary_erosion(true_mask)
    if not np.any(pred_surface) or not np.any(true_surface):
        return np.nan

    dt_pred = distance_transform_edt(~true_surface)
    dt_true = distance_transform_edt(~pred_surface)
    all_distances = np.concatenate([dt_pred[pred_surface], dt_true[true_surface]])
    return float(np.percentile(all_distances, 95))


def _collect_fold_predictions(results_dir: Path, n_folds: int):
    """Return [(case_id, pred_path, fold_idx)] across all folds' validation dirs."""
    preds = []
    seen = {}
    for fold in range(n_folds):
        val_dir = results_dir / f"fold_{fold}" / "validation"
        if not val_dir.is_dir():
            console.print(f"[yellow]No validation dir for fold {fold}: {val_dir}[/yellow]")
            continue
        for pred_file in sorted(val_dir.glob("*.nii.gz")):
            case_id = pred_file.name[: -len(".nii.gz")]
            if case_id in seen:
                console.print(
                    f"[yellow]{case_id} appears in folds {seen[case_id]} and {fold} "
                    f"(split overlap?) — keeping fold {seen[case_id]}[/yellow]"
                )
                continue
            seen[case_id] = fold
            preds.append((case_id, pred_file, fold))
    return preds


def _gt_path(data_paths, case_id: str):
    """Locate a case's seg across one or several pooled source dirs."""
    for dp in data_paths:
        case_dir = Path(dp) / case_id
        segs = list(case_dir.glob("*-seg.nii.gz")) if case_dir.exists() else []
        if segs:
            return segs[0]
    return None


def evaluate_kfold(results_dir, data_dir, output_dir, n_folds=5, visualize=True,
                   postprocess=False, postprocessing_json=None):
    results_path = Path(results_dir).expanduser().resolve()
    # Pool one or several source dirs, exactly as conversion did.
    data_dirs = data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]
    data_paths = [str(Path(d).expanduser().resolve()) for d in data_dirs]
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold cyan]Exp 19 — Native nnU-Net v2 ResEnc 5-fold OOF Evaluation[/bold cyan]\n"
        f"[dim]Results:      {results_path}\nGround truth: {data_paths}[/dim]",
        border_style="bright_blue",
    ))

    preds = _collect_fold_predictions(results_path, n_folds)
    if not preds:
        console.print(f"[red]No fold validation predictions found under {results_path}[/red]")
        return None
    console.print(f"\n[bold]Found {len(preds)} out-of-fold predictions across {n_folds} folds.[/bold]")

    # nnU-Net-style post-processing: determine on the OOF set (this IS the validation
    # set, so it's the CV metric nnU-Net reports), or apply a provided ops file.
    pp_ops = None
    if postprocessing_json:
        import json
        with open(postprocessing_json) as f:
            pp_ops = json.load(f).get("operations", [])
        console.print(f"[cyan]Applying provided post-processing ops: {pp_ops}[/cyan]")
    elif postprocess:
        from src.evaluation.nnunet_postprocessing import determine_postprocessing_streaming
        # build (pred_path, gt_path) items over cases that have GT
        items = []
        for case_id, pred_file, _ in preds:
            gp = _gt_path(data_paths, case_id)
            if gp is not None:
                items.append((pred_file, gp))
        # infer scheme from the data (BraTS 2024 -> 5 classes)
        num_classes = 5
        foreground = list(range(1, num_classes))
        regions_pp = {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3], "RC": [4]}
        regions_pp = {k: v for k, v in regions_pp.items() if all(l < num_classes for l in v)}

        def _load(it):
            pf, gf = it
            return (nib.load(str(pf)).get_fdata().astype(np.uint8),
                    nib.load(str(gf)).get_fdata().astype(np.uint8))

        console.print(f"\n[bold]Determining post-processing on {len(items)} OOF cases "
                      f"(streaming, multi-pass)...[/bold]")
        n_passes = 1 + len(foreground) + 1  # baseline + per-candidate
        bar = tqdm(total=len(items) * n_passes, desc="Determining PP")
        pp_ops, pp_report = determine_postprocessing_streaming(
            items, _load, foreground, regions_pp, on_case=lambda: bar.update(1)
        )
        bar.close()
        with open(output_path / "postprocessing.json", "w") as f:
            import json
            json.dump({"operations": pp_ops, "regions": regions_pp,
                       "num_classes": num_classes, "report": pp_report}, f, indent=2)
        console.print(f"[green]Determined ops {pp_ops} — baseline "
                      f"{pp_report['baseline_dice']:.4f} -> {pp_report['final_dice']:.4f} "
                      f"(gain {pp_report['gain']:+.4f}); saved postprocessing.json[/green]")

    if pp_ops:
        from src.evaluation.nnunet_postprocessing import apply_postprocessing

    records, vis_data = [], []
    for case_id, pred_file, fold in tqdm(preds, desc="Computing metrics"):
        pred_data = nib.load(str(pred_file)).get_fdata().astype(np.uint8)
        if pp_ops:
            pred_data = apply_postprocessing(pred_data, pp_ops)

        seg_file = _gt_path(data_paths, case_id)
        seg_files = [seg_file] if seg_file is not None else []
        if not seg_files:
            console.print(f"[yellow]No ground truth for {case_id}, skipping[/yellow]")
            continue

        gt_data = nib.load(str(seg_files[0])).get_fdata().astype(np.uint8)
        gt_remapped = np.zeros_like(gt_data)
        for src, dst in LABEL_REMAP.items():
            gt_remapped[gt_data == src] = dst

        record = {"case_id": case_id, "fold": fold}
        for class_idx, class_name in CLASS_NAMES.items():
            pred_mask = pred_data == class_idx
            true_mask = gt_remapped == class_idx
            record[f"dice_{class_name}"] = _dice(pred_mask, true_mask)
            try:
                record[f"hd95_{class_name}"] = _hausdorff95(pred_mask, true_mask)
            except Exception:
                record[f"hd95_{class_name}"] = np.nan
        for region_name, label_indices in REGIONS.items():
            pred_region = np.isin(pred_data, label_indices)
            true_region = np.isin(gt_remapped, label_indices)
            record[f"dice_{region_name}"] = _dice(pred_region, true_region)
            record[f"vol_pred_{region_name}"] = int(pred_region.sum())
            record[f"vol_true_{region_name}"] = int(true_region.sum())
        records.append(record)
        if visualize:
            vis_data.append((case_id, record.copy()))

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(output_path / "case_metrics.csv", index=False)
    console.print(f"[green]Saved metrics to {output_path / 'case_metrics.csv'}[/green]")

    # --- Per-region summary (mean over all OOF cases = the 5-fold CV number) ---
    region_table = Table(title="Exp 19 — Per-Region 5-fold OOF Dice", style="bold magenta")
    region_table.add_column("Region", style="bold")
    for col in ("Mean", "Std", "Median"):
        region_table.add_column(f"{col} Dice", justify="right")
    for region in REGIONS:
        col = f"dice_{region}"
        if col in metrics_df.columns:
            region_table.add_row(
                region,
                f"{metrics_df[col].mean():.4f}",
                f"{metrics_df[col].std():.4f}",
                f"{metrics_df[col].median():.4f}",
            )
    console.print(region_table)

    # --- Per-fold breakdown (sanity: spread across folds) ---
    fold_table = Table(title="Per-Fold Mean Dice", style="bold green")
    fold_table.add_column("Fold", style="bold")
    for r in REGIONS:
        fold_table.add_column(r, justify="right")
    fold_table.add_column("N", justify="right")
    for fold in sorted(metrics_df["fold"].unique()):
        sub = metrics_df[metrics_df["fold"] == fold]
        fold_table.add_row(
            str(fold),
            *[f"{sub[f'dice_{r}'].mean():.4f}" for r in REGIONS],
            str(len(sub)),
        )
    console.print(fold_table)

    # --- Failure analysis + distributions (shared code) ---
    eval_config = {"evaluation": {
        "regions": REGIONS,
        "failure_dice_threshold": 0.5,
        "small_tumor_volume_threshold": 500,
        "num_failure_cases": 10,
    }}
    console.print("\n[bold]Failure Analysis:[/bold]")
    print_failure_summary(identify_failure_cases(metrics_df, eval_config))
    if visualize:
        plot_metrics_distributions(metrics_df, eval_config, str(output_path))

    # --- config.yaml so analyze_failures.py can load this as a run ---
    fake_config = {
        "model": {"name": "nnunet_v2_native_resenc"},
        "data": {"train_dir": data_paths, "n_folds": n_folds, "split_seed": 42},
        "native": {"results_dir": str(results_path), "evaluate_on": "oof_validation"},
        "evaluation": eval_config["evaluation"],
    }
    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(fake_config, f)

    console.print(f"\n[bold green]Exp 19 evaluation complete. Results in: {output_path}[/bold green]")
    console.print("[dim]Compare with the custom-loop exp18 run:[/dim]")
    console.print(f"  python analyze_failures.py --run_dirs {output_path} runs/<exp18_eval> --compare")
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate native nnU-Net 5-fold OOF predictions")
    parser.add_argument("--results_dir", required=True,
                        help="nnU-Net trainer output dir (…__PLANS__CONFIG) containing fold_*/validation/")
    parser.add_argument("--data_dir", nargs="+", default=["../Brats2024/training_data1_v2"],
                        help="Original BraTS data dir(s) for ground truth. Pass the SAME "
                             "dir(s) used for conversion so the test split matches.")
    parser.add_argument("--output_dir", default="runs/exp19_nnunet_native_resenc_eval",
                        help="Output run directory")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--postprocess", action="store_true",
                        help="Determine nnU-Net-style post-processing on the OOF set and report with it")
    parser.add_argument("--postprocessing_json",
                        help="Apply a previously determined postprocessing.json instead of determining")
    args = parser.parse_args()

    evaluate_kfold(
        args.results_dir, args.data_dir, args.output_dir,
        n_folds=args.n_folds, visualize=not args.no_visualize,
        postprocess=args.postprocess, postprocessing_json=args.postprocessing_json,
    )
