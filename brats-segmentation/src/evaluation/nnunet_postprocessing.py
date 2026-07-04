"""nnU-Net-style connected-component post-processing (data-driven, not fixed thresholds).

nnU-Net does NOT hardcode component thresholds. Instead it runs a *determine* step on
the validation (out-of-fold) predictions:

  1. Measure baseline mean region Dice.
  2. Greedily test candidate operations in a fixed order:
        a. "remove all but largest component" over ALL foreground merged (1 tumor blob)
        b. then the same per individual foreground label
     A candidate is KEPT only if it improves mean validation Dice; otherwise discarded.
  3. The accepted operations are saved and later *applied* verbatim to the test set.

This matches `nnUNetv2_determine_postprocessing` / `apply_postprocessing` behavior for
label-based predictions (the custom-loop softmax case). Operates on integer label
volumes (0=background, 1..K-1 foreground), so it serves both BraTS 2024 (0..4) and 2023.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import label as cc_label


# ── core ops ───────────────────────────────────────────────────────────────────

def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Boolean mask of the single largest connected component in `binary`."""
    labeled, n = cc_label(binary)
    if n <= 1:
        return binary.astype(bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    return labeled == int(sizes.argmax())


def remove_all_but_largest(seg: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Keep only the largest connected component over the union of `labels`.

    Foreground voxels (in `labels`) that fall outside that component are set to 0.
    """
    seg = seg.copy()
    mask = np.isin(seg, list(labels))
    if not mask.any():
        return seg
    largest = _largest_component(mask)
    seg[mask & ~largest] = 0
    return seg


def apply_postprocessing(seg: np.ndarray, ops: List[Sequence[int]]) -> np.ndarray:
    """Apply a determined operation list (each entry = a label set) in order."""
    for label_set in ops:
        seg = remove_all_but_largest(seg, label_set)
    return seg


# ── metric ───────────────────────────────────────────────────────────────────

def _region_dice(pred: np.ndarray, gt: np.ndarray, labels: Sequence[int]) -> float:
    p = np.isin(pred, list(labels))
    g = np.isin(gt, list(labels))
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0  # both empty -> perfect (BraTS convention)
    return float(2.0 * (p & g).sum() / total)


def mean_region_dice(preds, gts, regions: Dict[str, Sequence[int]]) -> float:
    """Mean over cases of the mean region Dice (ET/TC/WT/...)."""
    per_case = [
        np.mean([_region_dice(p, g, lbls) for lbls in regions.values()])
        for p, g in zip(preds, gts)
    ]
    return float(np.mean(per_case)) if per_case else 0.0


# ── determine ──────────────────────────────────────────────────────────────────

def determine_postprocessing(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    foreground_labels: Sequence[int],
    regions: Dict[str, Sequence[int]],
    eps: float = 1e-4,
) -> Tuple[List[List[int]], dict]:
    """Greedily select connected-component ops that improve validation mean Dice.

    Returns:
        (ops, report) where ops is the ordered list of accepted label-sets and
        report holds baseline/final Dice and the per-candidate decision log.
    """
    baseline = mean_region_dice(preds, gts, regions)
    best = baseline
    current = preds
    ops: List[List[int]] = []
    log = []

    fg = tuple(sorted(int(c) for c in foreground_labels))
    candidates: List[Tuple[int, ...]] = [fg] + [(c,) for c in fg]

    for cand in candidates:
        trial = [remove_all_but_largest(p, cand) for p in current]
        score = mean_region_dice(trial, gts, regions)
        accept = score > best + eps
        log.append({"op": f"keep_largest{cand}", "dice": round(score, 5), "accepted": accept})
        if accept:
            current, best = trial, score
            ops.append(list(cand))

    report = {
        "baseline_dice": round(baseline, 5),
        "final_dice": round(best, 5),
        "gain": round(best - baseline, 5),
        "accepted_ops": ops,
        "log": log,
    }
    return ops, report


def determine_postprocessing_streaming(
    items,
    load_fn,
    foreground_labels: Sequence[int],
    regions: Dict[str, Sequence[int]],
    eps: float = 1e-4,
    on_case=None,
) -> Tuple[List[List[int]], dict]:
    """Memory-bounded determination: loads one case at a time (multi-pass).

    Identical greedy logic to ``determine_postprocessing`` but never holds the whole
    prediction set in memory — for the OOF set (up to ~1221 cases) it re-reads each
    case per candidate via ``load_fn(item) -> (pred_np, gt_np)``. ``on_case`` (if given)
    is called once per loaded case for progress reporting.
    """
    fg = tuple(sorted(int(c) for c in foreground_labels))
    candidates: List[Tuple[int, ...]] = [fg] + [(c,) for c in fg]

    def eval_with(ops_so_far: List[List[int]]) -> float:
        dices = []
        for it in items:
            pred, gt = load_fn(it)
            for op in ops_so_far:
                pred = remove_all_but_largest(pred, op)
            dices.append(np.mean([_region_dice(pred, gt, lbls) for lbls in regions.values()]))
            if on_case:
                on_case()
        return float(np.mean(dices)) if dices else 0.0

    baseline = eval_with([])
    best = baseline
    ops: List[List[int]] = []
    log = []
    for cand in candidates:
        score = eval_with(ops + [list(cand)])
        accept = score > best + eps
        log.append({"op": f"keep_largest{cand}", "dice": round(score, 5), "accepted": accept})
        if accept:
            ops.append(list(cand))
            best = score

    report = {
        "baseline_dice": round(baseline, 5),
        "final_dice": round(best, 5),
        "gain": round(best - baseline, 5),
        "accepted_ops": ops,
        "log": log,
    }
    return ops, report
