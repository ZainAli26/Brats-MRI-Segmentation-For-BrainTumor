#!/usr/bin/env python3
"""Score a prediction set lesion-wise AND voxel-wise, under three post-processing variants.

WARNING (2026-08-17): everything here is scored with the IN-HOUSE lesion-wise convention
(src/evaluation/lesionwise.py: uniform dil 3, GT floor 50, no pred pruning, no lesion
merging), NOT the official BraTS 2024 metric. The official-tool cross-check showed the two
disagree hard on exactly what this script measures — the in-house PP gains largely vanish
(NETC even reverses) under src/evaluation/lesionwise_official.py, and the threshold
defaults below are the in-house-determined values now superseded pending the official
re-sweep (nnunet_native/sweep_official_pp.py). Use this script for run-vs-run deltas under
a fixed convention; do NOT treat --save_labelmap_dir output as submission-final until the
thresholds are re-determined officially.

The sweep (sweep_small_component_pp.py) determined small-component thresholds by filtering
each REGION MASK at scoring time. A real submission cannot do that: it ships one label map,
and removing a small NETC (label 1) component also changes the TC and WT masks built from
it. That interaction was never measured. This script scores all three variants in one pass
so they can be compared case-by-case:

  raw       — predictions as they come out of the ensemble.
  region    — per-region-mask removal at scoring time, exactly as the sweep did.
              The upper bound the sweep's numbers refer to.
  labelmap  — per-label removal applied to the label map itself (the submission-realistic
              operation), regions rebuilt afterwards.

Thresholds default to the CV-determined, cross-fit-validated values (2026-08-17):
N=100 for the composite regions, N=50 for the raw labels; label-map removal uses the
raw-label thresholds for labels 1/2 and the composite ones for 3/4.

Works on a flat prediction directory (--pred_dir, e.g. the held-out test ensemble) or on
out-of-fold CV predictions collected across fold_*/validation (--results_dir). Determination
of thresholds stays on CV — running this on test only CONFIRMS numbers chosen there
(CLAUDE.md).

    OMP_NUM_THREADS=1 python nnunet_native/score_lesionwise.py \
        --results_dir <...>/nnUNetTrainer_750epochs__nnUNetResEncUNetPlans_11G__3d_fullres \
        --data_dir /data/Brats2024/training_all \
        --output_dir runs/exp19_cv_lesionwise_variants --num_processes 12
"""
from __future__ import annotations

import argparse
import json
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
from src.evaluation.nnunet_postprocessing import remove_small_components  # noqa: E402

REGIONS = {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3], "RC": [4], "NETC": [1], "SNFH": [2]}
COMPOSITE = ["ET", "TC", "WT", "RC"]
DEFAULT_REGION_THRESHOLDS = "ET:100,TC:100,WT:100,RC:100,NETC:50,SNFH:50"
DEFAULT_LABEL_THRESHOLDS = "1:50,2:50,3:100,4:100"
VARIANTS = ("raw", "region", "labelmap")

_G = {}


def _init_worker(cfg):
    """Pool initializer: explicit state hand-off so workers do not depend on the fork
    start method (spawn/forkserver re-import the module and would see an empty _G)."""
    _G.update(cfg)


def _json_safe(obj):
    """NaN -> None recursively, so the JSON artefacts stay strict-parser valid."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _voxel_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return np.nan
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def _filter_small(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1 or not mask.any():
        return mask
    lab, n = cc_label(mask, structure=generate_binary_structure(mask.ndim, mask.ndim))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.flatnonzero(sizes >= min_voxels)
    return np.isin(lab, keep) if keep.size else np.zeros_like(mask)


def _score_case(args):
    case_id, pred_path, gt_path = args
    region_t, label_t, do_hd95, save_dir = (
        _G["region_t"], _G["label_t"], _G["hd95"], _G["save_dir"])
    try:
        img = nib.load(str(pred_path))
        pred = img.get_fdata().astype(np.uint8)
        gt = nib.load(str(gt_path)).get_fdata().astype(np.uint8)
    except Exception as e:                      # a corrupt file must not kill the run
        return {"case_id": case_id, "error": f"{type(e).__name__}: {e}"}

    pp = pred
    for lbl, t in label_t.items():
        pp = remove_small_components(pp, [lbl], t)
    if save_dir is not None:
        # nib stores in the HEADER's declared dtype, not the array's — force uint8 so the
        # shipped files do not inherit a float32 dtype from whatever wrote the inputs.
        hdr = img.header.copy()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(pp, img.affine, hdr), str(save_dir / pred_path.name))

    rec = {"case_id": case_id}
    for region, labels in REGIONS.items():
        gt_mask = np.isin(gt, labels)
        masks = {
            "raw": np.isin(pred, labels),
            "labelmap": np.isin(pp, labels),
        }
        masks["region"] = _filter_small(masks["raw"], region_t[region])
        for variant in VARIANTS:
            lw = lesionwise_region(masks[variant], gt_mask, compute_hd95=do_hd95)
            rec[f"{region}|{variant}|lw_dice"] = lw["dice"]
            rec[f"{region}|{variant}|vox_dice"] = _voxel_dice(masks[variant], gt_mask)
            rec[f"{region}|{variant}|n_fp"] = lw["n_fp"]
            if do_hd95:
                rec[f"{region}|{variant}|lw_hd95"] = lw["hd95"]
    return rec


def _collect_oof(results_dir: Path, data_dir: Path, n_folds: int):
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


def _collect_flat(pred_dir: Path, data_dir: Path):
    items = []
    for p in sorted(pred_dir.glob("*.nii.gz")):
        cid = p.name[: -len(".nii.gz")]
        gt = list((data_dir / cid).glob("*-seg.nii.gz"))
        if gt:
            items.append((cid, p, gt[0]))
    return items


def _parse_thresholds(spec: str, cast_key):
    return {cast_key(k): int(v) for k, v in (part.split(":") for part in spec.split(","))}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred_dir", help="flat directory of prediction NIfTIs (e.g. test ensemble)")
    src.add_argument("--results_dir", help="nnU-Net results dir; collects fold_*/validation OOF")
    ap.add_argument("--data_dir", default="/data/Brats2024/training_all")
    ap.add_argument("--output_dir", default="runs/lesionwise_variants")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--num_processes", type=int, default=12)
    ap.add_argument("--region_thresholds", default=DEFAULT_REGION_THRESHOLDS)
    ap.add_argument("--label_thresholds", default=DEFAULT_LABEL_THRESHOLDS)
    ap.add_argument("--hd95", action="store_true", help="also compute lesion-wise HD95 (slow)")
    ap.add_argument("--limit", type=int, help="score only the first N cases (smoke test)")
    ap.add_argument("--save_labelmap_dir",
                    help="also write the label-map-post-processed predictions here "
                         "(the files a submission would ship)")
    args = ap.parse_args()

    _G["region_t"] = _parse_thresholds(args.region_thresholds, str)
    _G["label_t"] = _parse_thresholds(args.label_thresholds, int)
    _G["hd95"] = args.hd95
    _G["save_dir"] = None
    if args.save_labelmap_dir:
        _G["save_dir"] = Path(args.save_labelmap_dir)
        _G["save_dir"].mkdir(parents=True, exist_ok=True)
    missing = [r for r in REGIONS if r not in _G["region_t"]]
    if missing:
        ap.error(f"--region_thresholds missing regions: {missing}")

    data_dir = Path(args.data_dir)
    if args.pred_dir:
        items = _collect_flat(Path(args.pred_dir), data_dir)
    else:
        items = _collect_oof(Path(args.results_dir), data_dir, args.n_folds)
    if args.limit is not None:
        items = items[: args.limit]
    print(f"{len(items)} cases; region thresholds {_G['region_t']}; "
          f"label-map thresholds {_G['label_t']}; {args.num_processes} processes; "
          f"hd95={args.hd95}")

    with Pool(args.num_processes, initializer=_init_worker, initargs=(dict(_G),)) as pool:
        records = list(tqdm(pool.imap_unordered(_score_case, items, chunksize=4),
                            total=len(items), desc="scoring"))

    errs = [r for r in records if "error" in r]
    records = [r for r in records if "error" not in r]
    if errs:
        print(f"WARNING: {len(errs)} cases failed, e.g. {errs[0]}")

    # ---- aggregate: nanmean across cases, per region per variant ----
    summary = {}
    metrics = ["lw_dice", "vox_dice", "n_fp"] + (["lw_hd95"] if args.hd95 else [])
    for region in REGIONS:
        summary[region] = {}
        for variant in VARIANTS:
            for metric in metrics:
                vals = np.array([r.get(f"{region}|{variant}|{metric}", np.nan)
                                 for r in records], dtype=float)
                summary[region].setdefault(variant, {})[metric] = (
                    float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan"))

    def _mean(regions, variant, metric):
        return float(np.nanmean([summary[r][variant][metric] for r in regions]))

    means = {v: {"composite4": _mean(COMPOSITE, v, "lw_dice"),
                 "all6": _mean(REGIONS, v, "lw_dice"),
                 "composite4_vox": _mean(COMPOSITE, v, "vox_dice")}
             for v in VARIANTS}

    # ---- paired tests: does the label-map variant lose anything vs the sweep's upper
    # bound, and does it still beat raw? NOTE: "MEAN6_PER_CASE" is the mean over cases of
    # each case's nanmean across regions — a paired-testable statistic, but NOT the same
    # aggregation as the report's "all 6" column (mean of per-region means, each region
    # averaged over its own present-case subset). With NETC/RC absent in many cases the
    # two weight cases differently; do not reconcile one against the other. ----
    from scipy.stats import wilcoxon
    sig = {}
    for a, b in (("raw", "labelmap"), ("region", "labelmap")):
        sig[f"{a}_vs_{b}"] = {}
        for region in list(REGIONS) + ["MEAN6_PER_CASE"]:
            def per_case(variant, region=region):
                if region == "MEAN6_PER_CASE":
                    return np.array([np.nanmean([r.get(f"{rg}|{variant}|lw_dice", np.nan)
                                                 for rg in REGIONS]) for r in records])
                return np.array([r.get(f"{region}|{variant}|lw_dice", np.nan)
                                 for r in records])
            x, y = per_case(a), per_case(b)
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() > 10 and np.any(x[m] != y[m]):
                stat, p = wilcoxon(x[m], y[m])
                sig[f"{a}_vs_{b}"][region] = {
                    "n": int(m.sum()), "mean_delta": float(np.mean(y[m] - x[m])),
                    "p_value": float(p)}
            else:
                sig[f"{a}_vs_{b}"][region] = {
                    "n": int(m.sum()), "mean_delta": float(np.mean(y[m] - x[m])) if m.any() else 0.0,
                    "p_value": float("nan")}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(_json_safe({"n_cases": len(records),
                          "region_thresholds": _G["region_t"],
                          "label_thresholds": {str(k): v for k, v in _G["label_t"].items()},
                          "per_region": summary, "means": means, "significance": sig}),
              open(out / "summary.json", "w"), indent=2)
    # per_case keeps literal NaN: it is meaningful there and its consumers are Python.
    json.dump(records, open(out / "per_case.json", "w"))

    # ---- report ----
    print(f"\n{'variant':>9}{'lw mean (4 comp)':>18}{'lw mean (all 6)':>17}"
          f"{'vox mean (4 comp)':>19}   per-region lesion-wise")
    print("-" * 118)
    for v in VARIANTS:
        pr = "  ".join(f"{r} {summary[r][v]['lw_dice']:.4f}" for r in REGIONS)
        print(f"{v:>9}{means[v]['composite4']:>18.4f}{means[v]['all6']:>17.4f}"
              f"{means[v]['composite4_vox']:>19.4f}   {pr}")
    for pair, table in sig.items():
        print(f"\npaired Wilcoxon, {pair.replace('_', ' ')}:")
        for k, v in table.items():
            print(f"  {k:<6} n={v['n']:>5}  delta {v['mean_delta']:+.4f}  p={v['p_value']:.4g}")
    print(f"\nwrote {out}/summary.json and per_case.json")


if __name__ == "__main__":
    main()
