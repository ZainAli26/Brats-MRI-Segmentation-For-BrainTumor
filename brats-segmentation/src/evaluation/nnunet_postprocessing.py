"""nnU-Net v2 post-processing, reimplemented faithfully (data-driven, not fixed thresholds).

nnU-Net does NOT hardcode component thresholds and never removes *small* components,
fills holes, or suppresses tiny ET. Its only operation is "remove all but the largest
connected component", and whether to apply it is *decided* on the out-of-fold validation
predictions by `nnUNetv2_determine_postprocessing`:

  1. Score the raw validation predictions.
  2. Candidate A — keep only the largest component over ALL foreground labels merged.
     Accept iff ``foreground_mean`` Dice improves **and no individual label gets worse**
     (nnU-Net's defensive guard).
  3. Candidates B.. — keep only the largest component of each individual label, applied
     on top of whatever was accepted so far. Accept iff **that label's own** Dice improves.
  4. The accepted ops are saved and later applied verbatim to the test set.

This module matches ``nnunetv2/postprocessing/remove_connected_components.py`` on the
details that change the decision:

* **26-connectivity.** nnU-Net goes through ``skimage.measure.label(connectivity=None)``,
  which defaults to ``ndim`` — full 26-neighbour connectivity in 3D. ``scipy.ndimage.label``
  defaults to 6, which fragments blobs and makes "keep largest" delete far more.
* **Dice convention.** ``tp + fp + fn == 0`` (both masks empty) → NaN, *excluded* from the
  average, not scored as 1.0.
* **Aggregation order.** Per label: mean across cases first (``nanmean``), then mean across
  labels. Not per-case-mean-then-mean-across-cases — with NaNs present the two differ.
* **Acceptance.** Strict ``>``, no epsilon, plus the per-label non-degradation guard on the
  merged-foreground candidate.

Operates on integer label volumes (0 = background, 1..K-1 foreground), so it serves both
BraTS 2024 (0..4) and BraTS 2023 (0..3).

``criterion``
-------------
nnU-Net selects on the *labels* it was trained on. For BraTS the reported score is over the
overlapping regions (ET/TC/WT/RC) instead, so ``criterion="regions"`` selects on those.
Default is ``"labels"`` (nnU-Net-faithful); both scores are always reported so the choice is
visible. Region-based selection is what nnU-Net itself would do had it been trained with
region-based targets — it is not a deviation from the algorithm, only from our label-based
training setup.

One consequence to know before switching: the defensive guard can only protect what the
criterion scores. Under ``"regions"``, a label no region covers is invisible, so an op that
deletes a true lesion of that label is not vetoed — with ``"labels"`` it would be. For the
BraTS ET/TC/WT/RC set every label is covered, so the two agree there; the difference only
bites on a partial region set. That, plus fidelity, is why ``"labels"`` is the default.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as cc_label


# ── core ops ───────────────────────────────────────────────────────────────────

def _full_connectivity(ndim: int) -> np.ndarray:
    """Structuring element matching ``skimage.measure.label(connectivity=None)``.

    skimage defaults ``connectivity`` to the array's ndim, i.e. neighbours that touch at a
    corner count as connected (26 neighbours in 3D). scipy's default is 6.
    """
    return generate_binary_structure(ndim, ndim)


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest connected component(s), 26-connectivity in 3D.

    Ties are kept, not broken: acvl_utils' filter is ``[i for i, j in zip(ids, sizes)
    if j == max(sizes)]``, so every component matching the maximum size survives. Taking
    a single ``argmax`` instead silently deletes co-largest blobs.
    """
    binary = binary.astype(bool)
    labeled, n = cc_label(binary, structure=_full_connectivity(binary.ndim))
    if n <= 1:
        return binary
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    keep = np.flatnonzero(sizes == sizes.max())
    return np.isin(labeled, keep)


def remove_all_but_largest(seg: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Keep only the largest connected component over the union of ``labels``.

    Port of ``remove_all_but_largest_component_from_segmentation``: voxels in ``labels``
    that fall outside that component are set to background. Voxels of other labels are
    untouched, exactly as in nnU-Net.
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


# ── metric (nnU-Net's convention) ─────────────────────────────────────────────

def _dice(pred: np.ndarray, gt: np.ndarray, labels: Sequence[int]) -> float:
    """Dice for one label set. NaN when both masks are empty, as nnU-Net does.

    nnU-Net's ``compute_metrics`` sets Dice to NaN when ``tp + fp + fn == 0`` and drops it
    from the average via ``nanmean``. Scoring both-empty as 1.0 (the legacy BraTS
    convention) inflates the baseline and dilutes the measured gain of each candidate.
    """
    p = np.isin(pred, list(labels))
    g = np.isin(gt, list(labels))
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    if tp + fp + fn == 0:
        return float("nan")
    return float(2 * tp / (2 * tp + fp + fn))


def _key(labels: Sequence[int]) -> str:
    return ",".join(str(int(l)) for l in labels)


def summarize(per_case: List[Dict[str, float]], keys: Sequence[str]) -> Dict:
    """Aggregate like nnU-Net: nanmean per key across cases, then mean across keys."""
    with np.errstate(invalid="ignore"):
        mean = {}
        for k in keys:
            vals = [c[k] for c in per_case]
            mean[k] = float(np.nanmean(vals)) if any(not np.isnan(v) for v in vals) else float("nan")
    finite = [mean[k] for k in keys if not np.isnan(mean[k])]
    return {"mean": mean, "foreground_mean": float(np.mean(finite)) if finite else 0.0}


def _score_case(pred: np.ndarray, gt: np.ndarray, label_sets: Dict[str, Sequence[int]]) -> Dict[str, float]:
    return {k: _dice(pred, gt, v) for k, v in label_sets.items()}


def mean_region_dice(preds, gts, regions: Dict[str, Sequence[int]]) -> float:
    """Mean region Dice under nnU-Net aggregation (kept for reporting/back-compat)."""
    per_case = [_score_case(p, g, regions) for p, g in zip(preds, gts)]
    if not per_case:
        return 0.0
    return summarize(per_case, list(regions.keys()))["foreground_mean"]


# ── acceptance rules ──────────────────────────────────────────────────────────

def _accept_merged(base: Dict, trial: Dict, eps: float) -> bool:
    """nnU-Net's rule for the merged-foreground candidate: mean up, no label down.

    NaN comparisons are False in numpy, so a label that is NaN in either summary never
    vetoes the candidate — same as nnU-Net.
    """
    if not trial["foreground_mean"] > base["foreground_mean"] + eps:
        return False
    for k in base["mean"]:
        if trial["mean"][k] < base["mean"][k]:
            return False
    return True


def _accept_single(base: Dict, trial: Dict, key: str, eps: float) -> bool:
    """nnU-Net's rule for a per-label candidate: that label's own Dice must improve."""
    return bool(trial["mean"][key] > base["mean"][key] + eps)


def _build_candidates(
    foreground_labels: Sequence[int],
    regions: Optional[Dict[str, Sequence[int]]],
    criterion: str,
) -> Tuple[Dict[str, Sequence[int]], List[Tuple[str, List[int]]]]:
    """Return (scored label sets, ordered candidates as (selection_key, label_set)).

    Candidates always follow nnU-Net's order and definition — merged foreground first, then
    each individual label. Only the *metric* they are judged on depends on ``criterion``.
    """
    fg = [int(c) for c in sorted(set(int(c) for c in foreground_labels))]
    if criterion == "labels" or not regions:
        scored = {_key([c]): [c] for c in fg}
        per_label_key = {c: _key([c]) for c in fg}
    elif criterion == "regions":
        scored = {name: list(labels) for name, labels in regions.items()}
        # A per-label candidate is judged on the region that isolates it best: the smallest
        # scored region containing that label (e.g. label 3 -> ET, label 2 -> WT). A label no
        # region covers has no metric that isolates it, so it gets no candidate rather than
        # falling through to a rule meant for a different comparison.
        per_label_key = {}
        for c in fg:
            containing = [n for n, lbls in regions.items() if c in lbls]
            if containing:
                per_label_key[c] = min(containing, key=lambda n: len(regions[n]))
    else:
        raise ValueError(f"criterion must be 'labels' or 'regions', got {criterion!r}")

    # selection_key None marks the merged-foreground candidate, judged on foreground_mean.
    candidates: List[Tuple[Optional[str], List[int]]] = [(None, fg)]
    if len(fg) > 1:
        candidates += [(per_label_key[c], [c]) for c in fg if c in per_label_key]
    return scored, candidates


def num_determination_passes(
    foreground_labels: Sequence[int],
    regions: Optional[Dict[str, Sequence[int]]] = None,
    criterion: str = "labels",
) -> int:
    """How many full scoring passes determination will make (baseline + one per candidate).

    Callers sizing a progress bar should use this rather than assuming one candidate per
    label: the merged-foreground candidate is skipped when there is only one label, and
    under ``criterion="regions"`` a label no region covers gets no candidate at all.
    """
    _, candidates = _build_candidates(foreground_labels, regions, criterion)
    return 1 + len(candidates)


# ── determine ──────────────────────────────────────────────────────────────────

def determine_postprocessing(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    foreground_labels: Sequence[int],
    regions: Dict[str, Sequence[int]],
    eps: float = 0.0,
    criterion: str = "labels",
) -> Tuple[List[List[int]], dict]:
    """Select connected-component ops exactly as ``nnUNetv2_determine_postprocessing``.

    Args:
        preds/gts: matched lists of integer label volumes (use OUT-OF-FOLD validation
            predictions — determining on test data leaks).
        foreground_labels: e.g. ``[1, 2, 3, 4]`` for BraTS 2024.
        regions: BraTS scoring regions, e.g. ``{"ET": [3], "TC": [1, 3], ...}``. Always
            reported; used for selection only when ``criterion="regions"``.
        eps: required margin. nnU-Net uses strict ``>`` (0.0); raise it to demand a
            meaningful gain rather than noise.
        criterion: ``"labels"`` (nnU-Net-faithful) or ``"regions"`` (BraTS-scored).

    Returns:
        ``(ops, report)`` — ``ops`` is the ordered list of label-sets to feed
        ``apply_postprocessing``; ``report`` holds both metrics and the decision log.
    """
    items = list(zip(preds, gts))
    return _determine(
        items, lambda it: it, foreground_labels, regions, eps, criterion, on_case=None
    )


def determine_postprocessing_streaming(
    items,
    load_fn,
    foreground_labels: Sequence[int],
    regions: Dict[str, Sequence[int]],
    eps: float = 0.0,
    criterion: str = "labels",
    on_case=None,
) -> Tuple[List[List[int]], dict]:
    """Memory-bounded determination: loads one case at a time (multi-pass).

    Identical logic to ``determine_postprocessing`` but never holds the whole prediction
    set in memory — for a large OOF set it re-reads each case per candidate via
    ``load_fn(item) -> (pred_np, gt_np)``. ``on_case`` is called once per loaded case for
    progress reporting. Total loads = ``n_cases * (2 + n_foreground_labels)``.
    """
    return _determine(items, load_fn, foreground_labels, regions, eps, criterion, on_case)


def _determine(items, load_fn, foreground_labels, regions, eps, criterion, on_case):
    scored, candidates = _build_candidates(foreground_labels, regions, criterion)
    scored_keys = list(scored.keys())
    region_keys = list(regions.keys()) if regions else []
    # Always score the regions too, so the report shows the BraTS number either way.
    all_sets = dict(scored)
    for name, lbls in (regions or {}).items():
        all_sets.setdefault(name, list(lbls))

    def evaluate(ops: List[List[int]]) -> Tuple[Dict, Dict]:
        """One pass over the data with ``ops`` applied; returns (selection, regions)."""
        per_case = []
        for it in items:
            pred, gt = load_fn(it)
            pred = np.asarray(pred)
            gt = np.asarray(gt)
            gt = np.where(gt < 0, 0, gt)   # -1 marks outside-the-brain in the replica cache
            for op in ops:
                pred = remove_all_but_largest(pred, op)
            per_case.append(_score_case(pred, gt, all_sets))
            if on_case:
                on_case()
        sel = summarize(per_case, scored_keys)
        reg = summarize(per_case, region_keys) if region_keys else {"mean": {}, "foreground_mean": float("nan")}
        return sel, reg

    base_sel, base_reg = evaluate([])
    baseline_sel, baseline_reg = base_sel, base_reg
    ops: List[List[int]] = []
    log = []

    for sel_key, label_set in candidates:
        trial_sel, trial_reg = evaluate(ops + [label_set])
        if sel_key is None:
            accept = _accept_merged(base_sel, trial_sel, eps)
            judged_on = "foreground_mean"
        else:
            accept = _accept_single(base_sel, trial_sel, sel_key, eps)
            judged_on = sel_key
        log.append({
            "op": f"keep_largest({_key(label_set)})",
            "judged_on": judged_on,
            "selection_mean": _round(trial_sel["foreground_mean"]),
            "judged_value": _round(trial_sel["foreground_mean"] if sel_key is None
                                   else trial_sel["mean"][sel_key]),
            "judged_value_before": _round(base_sel["foreground_mean"] if sel_key is None
                                          else base_sel["mean"][sel_key]),
            "region_mean": _round(trial_reg["foreground_mean"]),
            "accepted": bool(accept),
        })
        if accept:
            ops.append([int(l) for l in label_set])
            base_sel, base_reg = trial_sel, trial_reg

    report = {
        "criterion": criterion,
        "accepted_ops": ops,
        "connectivity": "full (26 in 3D, matches nnU-Net/skimage)",
        # Back-compat keys: *_dice track the criterion that drove selection.
        "baseline_dice": _round(baseline_sel["foreground_mean"]),
        "final_dice": _round(base_sel["foreground_mean"]),
        "gain": _round(base_sel["foreground_mean"] - baseline_sel["foreground_mean"]),
        "baseline_per_key": {k: _round(v) for k, v in baseline_sel["mean"].items()},
        "final_per_key": {k: _round(v) for k, v in base_sel["mean"].items()},
        "baseline_region_dice": _round(baseline_reg["foreground_mean"]),
        "final_region_dice": _round(base_reg["foreground_mean"]),
        "region_gain": _round(base_reg["foreground_mean"] - baseline_reg["foreground_mean"]),
        "baseline_per_region": {k: _round(v) for k, v in baseline_reg["mean"].items()},
        "final_per_region": {k: _round(v) for k, v in base_reg["mean"].items()},
        "log": log,
    }
    return ops, report


def _round(v, nd: int = 5):
    v = float(v)
    return None if np.isnan(v) else round(v, nd)
