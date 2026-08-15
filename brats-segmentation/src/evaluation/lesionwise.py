"""Lesion-wise Dice and HD95, the metric BraTS 2023+ actually ranks on.

WHY THIS EXISTS
---------------
The repo's existing metric (`evaluate_nnunet_kfold.py:_dice`) is voxel-wise:
``2|A∩B| / (|A|+|B|)`` over a region mask. That is NOT what the BraTS challenge ranks on,
and the two disagree in a specific, important way:

  * voxel-wise: a 20-voxel false-positive speck changes the denominator by 20 out of
    ~19,000. Effectively invisible.
  * lesion-wise: that same speck is an unmatched predicted component and scores a fresh
    0.0, averaged in with equal weight to a 5,000-voxel lesion. Brutally expensive.

That difference is the whole reason small-component removal is worth testing: it only
pays under a lesion-wise metric. Measuring it voxel-wise gives a false negative.

DEFINITION IMPLEMENTED
----------------------
Per region, following the BraTS 2023 lesion-wise specification:

  1. Connected-component label the GT with full connectivity (26-neighbour in 3D).
  2. Dilate each GT lesion by ``dil_factor`` before matching, so a prediction that is
     close but not touching still counts as a hit, and so GT lesions fragmented by
     annotation noise are not double-counted.
  3. Drop GT lesions smaller than ``min_gt_voxels`` — the challenge excludes specks in
     the reference from scoring, otherwise annotation noise dominates the average.
  4. For each surviving GT lesion: Dice against the union of predicted components that
     overlap its dilated mask. No overlap => 0.0 (missed lesion).
  5. Each predicted component overlapping NO GT lesion => 0.0 (false-positive lesion).
  6. Lesion-wise Dice = mean over (matched lesions + missed lesions + FP lesions).

HD95 follows the same partition: per matched lesion pair; missed and false-positive
lesions contribute ``fp_fn_penalty`` (BraTS uses 374, the image diagonal) so the average
is defined even when a case has no matched lesions at all.

CALIBRATION WARNING
-------------------
This is implemented to the published spec, not vendored from the official evaluator, and
the two could differ at the margins (exact dilation structuring element, tie-breaking when
one prediction spans two GT lesions). ``dil_factor`` and ``min_gt_voxels`` are the two
parameters that move numbers most, so both are explicit rather than buried. Before any of
this goes into a paper, cross-check a handful of cases against the official BraTS tool.
Treat the DELTA between conditions as much more trustworthy than the absolute value: the
post-processing sweep compares runs computed the same way, so convention differences
largely cancel.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)
from scipy.ndimage import label as cc_label

# BraTS reference value: the diagonal of a 240x240x155 volume, used as the HD95 penalty
# for a lesion that is entirely missed or entirely spurious.
FP_FN_PENALTY = 374.0


def _full_connectivity(ndim: int) -> np.ndarray:
    """26-neighbour in 3D, matching the repo's existing post-processing convention."""
    return generate_binary_structure(ndim, ndim)


def _hd95_pair(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff between two binary masks, in voxels.

    Same construction as evaluate_nnunet_kfold.py:_hausdorff95 so the two metrics stay
    comparable; only the masks it is fed differ (one lesion pair, not the whole region).
    """
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if not pred_mask.any() or not gt_mask.any():
        return FP_FN_PENALTY

    pred_surface = pred_mask ^ binary_erosion(pred_mask)
    gt_surface = gt_mask ^ binary_erosion(gt_mask)
    # A single-voxel component erodes to nothing; its surface is the voxel itself.
    if not pred_surface.any():
        pred_surface = pred_mask
    if not gt_surface.any():
        gt_surface = gt_mask

    dt_to_gt = distance_transform_edt(~gt_surface)
    dt_to_pred = distance_transform_edt(~pred_surface)
    d = np.concatenate([dt_to_gt[pred_surface], dt_to_pred[gt_surface]])
    return float(np.percentile(d, 95))


def lesionwise_region(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    dil_factor: int = 3,
    min_gt_voxels: int = 50,
    compute_hd95: bool = True,
) -> Dict[str, float]:
    """Lesion-wise Dice/HD95 for ONE region (both inputs already binarised).

    Returns nan Dice when the region is absent from both prediction and reference, which
    matches how the voxel-wise path reports an absent region — so the two can be averaged
    with the same nan handling.
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if not gt_mask.any() and not pred_mask.any():
        return {"dice": np.nan, "hd95": np.nan, "n_gt": 0, "n_fp": 0, "n_matched": 0}

    struct = _full_connectivity(gt_mask.ndim)
    gt_lab, n_gt_raw = cc_label(gt_mask, structure=struct)
    pred_lab, n_pred = cc_label(pred_mask, structure=struct)

    # Reference lesions below the challenge's size floor are not scored at all: they are
    # neither credited if found nor penalised if missed.
    gt_ids = []
    for i in range(1, n_gt_raw + 1):
        if (gt_lab == i).sum() >= min_gt_voxels:
            gt_ids.append(i)

    dices: List[float] = []
    hd95s: List[float] = []
    matched_pred: set = set()
    n_matched = 0

    for gid in gt_ids:
        this_gt = gt_lab == gid
        # Dilate to decide MATCHING only. Dice itself is computed against the undilated
        # lesion, otherwise every score would be inflated by the dilation margin.
        search = binary_dilation(this_gt, structure=struct, iterations=dil_factor)
        hits = np.unique(pred_lab[search])
        hits = hits[hits != 0]

        if hits.size == 0:
            dices.append(0.0)                      # missed lesion
            if compute_hd95:
                hd95s.append(FP_FN_PENALTY)
            continue

        matched_pred.update(int(h) for h in hits)
        n_matched += 1
        this_pred = np.isin(pred_lab, hits)
        inter = np.logical_and(this_pred, this_gt).sum()
        denom = this_pred.sum() + this_gt.sum()
        dices.append(2.0 * inter / denom if denom else 0.0)
        if compute_hd95:
            hd95s.append(_hd95_pair(this_pred, this_gt))

    # Predicted components matching no scored GT lesion are false-positive lesions. This
    # is the term small-component removal is meant to attack.
    n_fp = 0
    for pid in range(1, n_pred + 1):
        if pid in matched_pred:
            continue
        n_fp += 1
        dices.append(0.0)
        if compute_hd95:
            hd95s.append(FP_FN_PENALTY)

    if not dices:
        # GT had only sub-threshold lesions and the prediction matched them all.
        return {"dice": np.nan, "hd95": np.nan, "n_gt": 0, "n_fp": 0, "n_matched": 0}

    return {
        "dice": float(np.mean(dices)),
        "hd95": float(np.mean(hd95s)) if compute_hd95 and hd95s else np.nan,
        "n_gt": len(gt_ids),
        "n_fp": n_fp,
        "n_matched": n_matched,
    }


def lesionwise_case(
    pred: np.ndarray,
    gt: np.ndarray,
    regions: Dict[str, Sequence[int]],
    dil_factor: int = 3,
    min_gt_voxels: int = 50,
    compute_hd95: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Lesion-wise metrics for every region of one case."""
    out = {}
    for name, labels in regions.items():
        out[name] = lesionwise_region(
            np.isin(pred, list(labels)), np.isin(gt, list(labels)),
            dil_factor=dil_factor, min_gt_voxels=min_gt_voxels, compute_hd95=compute_hd95,
        )
    return out
