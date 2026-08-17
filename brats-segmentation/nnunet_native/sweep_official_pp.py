#!/usr/bin/env python3
"""Re-determine small-component removal thresholds under the OFFICIAL BraTS 2024 GLI
lesion-wise metric (src/evaluation/lesionwise_official.py, verified 144/144 exact against
the official tool on 2026-08-17).

Why the in-house sweep result cannot be shipped: the official evaluator already deletes
predicted lesions <= 20 voxels (<= 10 ET) and merges dilation-touching components, so the
in-house N=100/N=50 thresholds — chosen against a metric where every speck and every
unmerged satellite scores a fresh 0 — mostly stopped paying (and hurt NETC) when scored
officially. This sweep chooses thresholds against the metric the challenge actually ranks
on. Determination on the out-of-fold CV set only, never test (CLAUDE.md).

Two modes:
  * marginal sweep (default): for each raw label independently, sweep the removal
    threshold and score only the tissues whose MASK that label participates in (AFFECTED).
    Cross-tissue partner maps are cached from the unmodified prediction — exact for the
    scored tissues, because sweeping label L leaves every other label's lesion map
    untouched. KNOWN LIMITATION: editing label L can also change OTHER tissues' scores
    through the cross-tissue combining (a deleted >20-voxel component may have been the
    touching bridge that merged two lesions of a partner tissue). The marginal curves do
    not see that, by construction; it is second-order (needs a >20-voxel component that is
    also a unique merge bridge) but not impossible.
  * --joint "1:0,2:50,3:0,4:0": score one explicit combination on all six tissues against
    baseline with NO caching — full preparation of the edited label map. This is the gate
    that catches what the marginal pass ignores: never ship a combo whose joint pass shows
    a tissue significantly hurt that the marginal curves showed flat.

NOTE: the per-case keys are '{tissue}|L{label}T{thr}|lw_dice' (baseline '{tissue}|0|...'),
which crossfit_lesionwise_threshold.py's numeric-threshold key scheme does not parse —
adapt it before cross-fitting this sweep's choice (it now aborts loudly on a key-scheme
mismatch instead of validating a degenerate {0} sweep).

    OMP_NUM_THREADS=1 python nnunet_native/sweep_official_pp.py \
        --results_dir <...>__3d_fullres --data_dir /data/Brats2024/training_all \
        --output_dir /workspace/lesionwise_eval/exp19_cv_official_sweep --num_processes 12
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as cc_label

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.lesionwise_official import (  # noqa: E402
    CROSS_PARTNERS, TISSUES, _combined_cc, _dil_factor, _prune, _relabel_by_pairs,
    _renumber, _score_tissue, _touching_pairs, _volume_floor, prepare_gt, prepare_pred)
from src.evaluation.nnunet_postprocessing import remove_small_components  # noqa: E402

_CC26 = generate_binary_structure(3, 3)

# Tissues whose lesion map changes when a given raw label is edited.
AFFECTED = {1: ["NETC", "TC", "WT"], 2: ["SNFH", "WT"], 3: ["ET", "TC", "WT"], 4: ["RC"]}
DEFAULT_THRESHOLDS = [10, 20, 30, 50, 75, 100, 150]

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


def _pred_maps_swept(tissue_masks, label, cached_base):
    """Maps for the affected tissues only, partners taken from the cached baseline.

    ``tissue_masks``: the affected tissues' boolean masks with the swept label's
    sub-threshold components already knocked out."""
    maps = dict(cached_base)
    for t in AFFECTED[label]:
        maps[t] = _prune(_combined_cc(tissue_masks[t], _dil_factor(t)), _volume_floor(t))
    out = {}
    for t in AFFECTED[label]:
        cur = maps[t]
        for p in CROSS_PARTNERS.get(t, []):
            cur = _relabel_by_pairs(cur, _touching_pairs(cur, maps[p]))
        out[t] = _renumber(cur)
    return out


def _score_case(args):
    case_id, pred_path, gt_path = args
    thresholds, joint = _G["thresholds"], _G["joint"]
    try:
        pred = nib.load(str(pred_path)).get_fdata().astype(np.int32)
        gt = nib.load(str(gt_path)).get_fdata().astype(np.int32)
    except Exception as e:
        return {"case_id": case_id, "error": f"{type(e).__name__}: {e}"}

    gt_maps = prepare_gt(gt)
    rec = {"case_id": case_id}

    cached_base, base_final = prepare_pred(pred, return_base=True)
    for t in TISSUES:
        s = _score_tissue(base_final[t], gt_maps[t], t)
        rec[f"{t}|0|lw_dice"] = s["dice"]
        rec[f"{t}|0|n_fp"] = s["n_fp"]
        rec[f"{t}|0|absent"] = s["absent"]

    if joint is not None:
        seg = pred
        for lbl, thr in joint.items():
            seg = remove_small_components(seg, [lbl], thr)
        final = prepare_pred(seg)
        for t in TISSUES:
            s = _score_tissue(final[t], gt_maps[t], t)
            rec[f"{t}|joint|lw_dice"] = s["dice"]
            rec[f"{t}|joint|n_fp"] = s["n_fp"]
        return rec

    # Label each raw label's components ONCE; every threshold is then a size-table lookup.
    # Thresholds that remove the identical component set (or nothing) reuse the previous
    # scores instead of recomputing — the dominant saving, since most cases have only a
    # couple of small components per label.
    base_masks = {t: np.isin(pred, TISSUES[t])
                  for t in {t for ts in AFFECTED.values() for t in ts}}
    for label in AFFECTED:
        lab, n = cc_label(pred == label, structure=_CC26)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        prev_doomed, prev_scores = None, None
        for thr in thresholds:
            doomed = frozenset(np.flatnonzero((sizes > 0) & (sizes < thr)).tolist())
            if doomed == prev_doomed:
                scores = prev_scores
            elif not doomed:
                scores = {t: {"dice": rec[f"{t}|0|lw_dice"], "n_fp": rec[f"{t}|0|n_fp"]}
                          for t in AFFECTED[label]}
            else:
                removed = np.isin(lab, list(doomed))
                masks = {t: base_masks[t] & ~removed for t in AFFECTED[label]}
                swept = _pred_maps_swept(masks, label, cached_base)
                scores = {t: _score_tissue(swept[t], gt_maps[t], t)
                          for t in AFFECTED[label]}
            prev_doomed, prev_scores = doomed, scores
            for t in AFFECTED[label]:
                rec[f"{t}|L{label}T{thr}|lw_dice"] = scores[t]["dice"]
                rec[f"{t}|L{label}T{thr}|n_fp"] = scores[t]["n_fp"]
    return rec


def _collect_oof(results_dir: Path, data_dir: Path, n_folds: int):
    items, seen = [], {}
    for fold in range(n_folds):
        vd = results_dir / f"fold_{fold}" / "validation"
        if not vd.is_dir():
            continue
        for p in sorted(vd.glob("*.nii.gz")):
            cid = p.name[: -len(".nii.gz")]
            if cid in seen:
                raise RuntimeError(f"{cid} in folds {seen[cid]} AND {fold} — not out-of-fold")
            seen[cid] = fold
            gt = list((data_dir / cid).glob("*-seg.nii.gz"))
            if gt:
                items.append((cid, p, gt[0]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--data_dir", default="/data/Brats2024/training_all")
    ap.add_argument("--output_dir", default="runs/official_pp_sweep")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--num_processes", type=int, default=12)
    ap.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    ap.add_argument("--joint", help="score ONE combo '1:0,2:50,3:0,4:0' instead of sweeping")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    _G["thresholds"] = [int(t) for t in args.thresholds.split(",")]
    _G["joint"] = None
    if args.joint:
        _G["joint"] = {int(k): int(v) for k, v in
                       (part.split(":") for part in args.joint.split(","))}
        _G["joint"] = {k: v for k, v in _G["joint"].items() if v > 1}

    items = _collect_oof(Path(args.results_dir), Path(args.data_dir), args.n_folds)
    if args.limit is not None:
        items = items[: args.limit]
    mode = f"joint {_G['joint']}" if _G["joint"] is not None else \
        f"marginal sweep, thresholds {_G['thresholds']}"
    print(f"{len(items)} out-of-fold cases; {mode}; {args.num_processes} processes")

    with Pool(args.num_processes, initializer=_init_worker, initargs=(dict(_G),)) as pool:
        records = list(tqdm(pool.imap_unordered(_score_case, items, chunksize=2),
                            total=len(items), desc="scoring"))
    errs = [r for r in records if "error" in r]
    records = [r for r in records if "error" not in r]
    if errs:
        print(f"WARNING: {len(errs)} cases failed, e.g. {errs[0]}")

    from scipy.stats import wilcoxon

    def col(key):
        return np.array([r.get(key, np.nan) for r in records], dtype=float)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"n_cases": len(records), "mode": mode}

    if _G["joint"] is not None:
        print(f"\n{'tissue':>7}{'baseline':>10}{'joint':>10}{'delta':>9}{'p':>12}")
        summary["tissues"] = {}
        for t in TISSUES:
            a, b = col(f"{t}|0|lw_dice"), col(f"{t}|joint|lw_dice")
            m = ~(np.isnan(a) | np.isnan(b))
            delta = float(np.mean(b[m] - a[m]))
            p = float(wilcoxon(a[m], b[m]).pvalue) if np.any(a[m] != b[m]) else float("nan")
            summary["tissues"][t] = {"baseline": float(np.mean(a[m])),
                                     "joint": float(np.mean(b[m])),
                                     "delta": delta, "p": p, "n": int(m.sum())}
            print(f"{t:>7}{np.mean(a[m]):>10.4f}{np.mean(b[m]):>10.4f}{delta:>+9.4f}{p:>12.3g}")
        base6 = np.nanmean([col(f"{t}|0|lw_dice") for t in TISSUES], axis=0)
        joint6 = np.nanmean([col(f"{t}|joint|lw_dice") for t in TISSUES], axis=0)
        summary["mean6"] = {"baseline": float(np.nanmean(base6)),
                            "joint": float(np.nanmean(joint6)),
                            "delta": float(np.nanmean(joint6 - base6)),
                            "p": float(wilcoxon(base6, joint6).pvalue)
                            if np.any(base6 != joint6) else float("nan")}
        m6 = summary["mean6"]
        print(f"{'MEAN6':>7}{m6['baseline']:>10.4f}{m6['joint']:>10.4f}"
              f"{m6['delta']:>+9.4f}{m6['p']:>12.3g}")
    else:
        summary["curves"] = {}
        for label, tissues in AFFECTED.items():
            print(f"\n--- removing label {label} components < N "
                  f"(official metric, mean over {len(records)} cases; "
                  f"** gain / xx harm at p<1e-3) ---")
            header = f"{'N':>5}" + "".join(f"{t:>18}" for t in tissues) + f"{'FPs':>16}"
            print(header)
            for thr in [0] + _G["thresholds"]:
                key = "0" if thr == 0 else f"L{label}T{thr}"
                row = f"{thr:>5}"
                for t in tissues:
                    d = col(f"{t}|{key}|lw_dice")
                    a = col(f"{t}|0|lw_dice")
                    m = ~(np.isnan(a) | np.isnan(d))
                    mean = float(np.mean(d[m]))
                    cell = {"mean": mean}
                    if thr == 0:
                        row += f"{mean:.4f}".rjust(18)
                    else:
                        delta = float(np.mean(d[m] - a[m]))
                        p = (float(wilcoxon(a[m], d[m]).pvalue)
                             if np.any(a[m] != d[m]) else float("nan"))
                        star = "**" if p < 1e-3 and delta > 0 else \
                               ("xx" if p < 1e-3 and delta < 0 else "  ")
                        row += f"{mean:.4f} {delta:+.4f}{star}".rjust(18)
                        cell.update(delta=delta, p=p)
                    summary["curves"].setdefault(f"L{label}", {}).setdefault(
                        str(thr), {})[t] = cell
                fp = col(f"{tissues[0]}|{key}|n_fp")
                row += f"{np.nanmean(fp):>10.2f} {tissues[0]}"
                print(row)

    json.dump(_json_safe(summary), open(out / "sweep.json", "w"), indent=2)
    # per_case keeps literal NaN: it is meaningful there and its consumers are Python.
    json.dump(records, open(out / "per_case.json", "w"))
    print(f"\nwrote {out}/sweep.json and per_case.json")


if __name__ == "__main__":
    main()
