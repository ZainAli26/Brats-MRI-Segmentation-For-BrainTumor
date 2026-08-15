#!/usr/bin/env python3
"""Sweep small-component removal thresholds on the out-of-fold predictions, scored BOTH
voxel-wise and lesion-wise.

Answers two questions the existing evaluation cannot:

  1. Where does our Dice sit on the metric BraTS actually ranks on? The repo's number
     (0.733 for exp19) is voxel-wise; the challenge is lesion-wise, which punishes
     false-positive components far harder.
  2. Does removing sub-N-voxel components help? nnU-Net's own post-processing search only
     considers "keep all but the largest", which was determined on this run and REJECTED
     (every region got worse). Size thresholding is the opposite operation and is untested.

Determination happens on the OUT-OF-FOLD CV set only, never on test (CLAUDE.md).

    OMP_NUM_THREADS=1 python nnunet_native/sweep_small_component_pp.py \
        --results_dir <...>/nnUNetTrainer_750epochs__nnUNetResEncUNetPlans_11G__3d_fullres \
        --data_dir /data/Brats2024/training_all \
        --output_dir runs/exp19_lesionwise_pp --num_processes 16

PERFORMANCE NOTE: connected components are labelled ONCE per case+region and the sweep
then filters by component size, rather than re-labelling per threshold. That turns
7 thresholds x 4 regions x 1468 cases from ~41k labelling passes into ~5.9k.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as cc_label
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.lesionwise import lesionwise_region  # noqa: E402

REGIONS = {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3], "RC": [4]}
DEFAULT_THRESHOLDS = [0, 10, 25, 50, 100, 250, 500]


def _voxel_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return np.nan          # absent from both: same convention as the voxel-wise path
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def _sized_components(mask: np.ndarray):
    """Label once, return (labeled array, per-component voxel counts)."""
    lab, n = cc_label(mask, structure=generate_binary_structure(mask.ndim, mask.ndim))
    if n == 0:
        return lab, np.zeros(1, dtype=np.int64)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab, sizes


def _score_case(args):
    case_id, pred_path, gt_path, thresholds, do_hd95 = args
    try:
        pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
        gt = nib.load(str(gt_path)).get_fdata().astype(np.uint8)
    except Exception as e:                      # a corrupt file must not kill the sweep
        return {"case_id": case_id, "error": f"{type(e).__name__}: {e}"}

    rec = {"case_id": case_id}
    for region, labels in REGIONS.items():
        gt_mask = np.isin(gt, labels)
        pred_mask = np.isin(pred, labels)
        lab, sizes = _sized_components(pred_mask)

        for t in thresholds:
            if t <= 1:
                pm = pred_mask
            else:
                keep = np.flatnonzero(sizes >= t)
                pm = np.isin(lab, keep) if keep.size else np.zeros_like(pred_mask)
            lw = lesionwise_region(pm, gt_mask, compute_hd95=do_hd95)
            rec[f"{region}|{t}|lw_dice"] = lw["dice"]
            rec[f"{region}|{t}|vox_dice"] = _voxel_dice(pm, gt_mask)
            rec[f"{region}|{t}|n_fp"] = lw["n_fp"]
            if do_hd95:
                rec[f"{region}|{t}|lw_hd95"] = lw["hd95"]
    return rec


def _collect(results_dir: Path, data_dir: Path, n_folds: int):
    """One prediction per case — the fold that held it out. No ensembling: that would leak."""
    items, seen = [], {}
    for fold in range(n_folds):
        vd = results_dir / f"fold_{fold}" / "validation"
        if not vd.is_dir():
            continue
        for p in sorted(vd.glob("*.nii.gz")):
            cid = p.name[: -len(".nii.gz")]
            if cid in seen:
                raise RuntimeError(
                    f"{cid} predicted by fold {seen[cid]} AND {fold} — not out-of-fold")
            seen[cid] = fold
            gt = list((data_dir / cid).glob("*-seg.nii.gz"))
            if gt:
                items.append((cid, p, gt[0]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--data_dir", default="/data/Brats2024/training_all")
    ap.add_argument("--output_dir", default="runs/lesionwise_pp")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--num_processes", type=int, default=16)
    ap.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    ap.add_argument("--hd95", action="store_true", help="also compute lesion-wise HD95 (slow)")
    ap.add_argument("--limit", type=int, help="score only the first N cases (smoke test)")
    args = ap.parse_args()

    thresholds = [int(t) for t in args.thresholds.split(",")]
    items = _collect(Path(args.results_dir), Path(args.data_dir), args.n_folds)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} out-of-fold cases; thresholds {thresholds}; "
          f"{args.num_processes} processes; hd95={args.hd95}")

    payload = [(c, p, g, thresholds, args.hd95) for c, p, g in items]
    with Pool(args.num_processes) as pool:
        records = list(tqdm(pool.imap_unordered(_score_case, payload, chunksize=4),
                            total=len(payload), desc="scoring"))

    errs = [r for r in records if "error" in r]
    records = [r for r in records if "error" not in r]
    if errs:
        print(f"WARNING: {len(errs)} cases failed, e.g. {errs[0]}")

    # ---- aggregate: nanmean across cases, per region per threshold ----
    summary = {}
    for region in REGIONS:
        summary[region] = {}
        for t in thresholds:
            for metric in (["lw_dice", "vox_dice", "n_fp"] + (["lw_hd95"] if args.hd95 else [])):
                vals = np.array([r.get(f"{region}|{t}|{metric}", np.nan) for r in records],
                                dtype=float)
                summary[region].setdefault(str(t), {})[metric] = (
                    float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan"))

    means = {str(t): float(np.nanmean([summary[r][str(t)]["lw_dice"] for r in REGIONS]))
             for t in thresholds}
    vox_means = {str(t): float(np.nanmean([summary[r][str(t)]["vox_dice"] for r in REGIONS]))
                 for t in thresholds}
    best_t = max(means, key=lambda k: means[k])

    # ---- paired significance test, baseline vs best, on the shared cases ----
    sig = {}
    if best_t != "0":
        from scipy.stats import wilcoxon
        for region in list(REGIONS) + ["MEAN"]:
            def per_case(t):
                if region == "MEAN":
                    return np.array([np.nanmean([r.get(f"{rg}|{t}|lw_dice", np.nan)
                                                 for rg in REGIONS]) for r in records])
                return np.array([r.get(f"{region}|{t}|lw_dice", np.nan) for r in records])
            a, b = per_case(0), per_case(int(best_t))
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() > 10 and np.any(a[m] != b[m]):
                stat, p = wilcoxon(a[m], b[m])
                sig[region] = {"n": int(m.sum()), "mean_delta": float(np.mean(b[m] - a[m])),
                               "p_value": float(p)}
            else:
                sig[region] = {"n": int(m.sum()), "mean_delta": 0.0, "p_value": float("nan")}

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump({"n_cases": len(records), "thresholds": thresholds, "per_region": summary,
               "lesionwise_region_mean": means, "voxelwise_region_mean": vox_means,
               "best_threshold": int(best_t), "significance_vs_baseline": sig},
              open(out / "sweep.json", "w"), indent=2)
    json.dump(records, open(out / "per_case.json", "w"))

    # ---- report ----
    print(f"\n{'thresh':>7}{'LESION-wise mean':>19}{'voxel-wise mean':>18}   per-region lesion-wise")
    print("-" * 95)
    for t in thresholds:
        pr = "  ".join(f"{r} {summary[r][str(t)]['lw_dice']:.4f}" for r in REGIONS)
        star = "  <- best" if str(t) == best_t else ""
        print(f"{t:>7}{means[str(t)]:>19.4f}{vox_means[str(t)]:>18.4f}   {pr}{star}")
    print(f"\nmean false-positive lesions per case (the term this attacks):")
    for r in REGIONS:
        print(f"  {r:<4}" + "  ".join(f"N={t}: {summary[r][str(t)]['n_fp']:.2f}" for t in thresholds))
    if sig:
        print(f"\npaired Wilcoxon, threshold {best_t} vs raw:")
        for k, v in sig.items():
            print(f"  {k:<5} n={v['n']:>5}  delta {v['mean_delta']:+.4f}  p={v['p_value']:.4g}")
    print(f"\nwrote {out}/sweep.json and per_case.json")


if __name__ == "__main__":
    main()
