"""Fast reimplementation of the OFFICIAL BraTS 2024 GLI lesion-wise Dice
(rachitsaluja/BraTS-2024-Metrics, metrics_GLI.py), bug-for-bug.

WHY THIS EXISTS
---------------
The in-house metric (lesionwise.py) was written to the published spec and agrees with the
official tool to 4 decimals on clean cases — but the official pipeline has five extra
behaviours that change absolute scores AND flip post-processing conclusions (measured
2026-08-17 on 12 OOF cases: the in-house +0.12..+0.22 PP deltas collapse to ~0 for
ET/TC/RC, go NEGATIVE for NETC, and survive only for WT/SNFH at ~+0.07):

  1. Predicted lesions of <= 20 voxels (<= 10 for ET) are DELETED before scoring, so small
     FP specks are already free under the official metric.
  2. Components whose ``dil_factor`` dilations touch are MERGED into one lesion (GT and
     pred both), so near-main-lesion satellites are absorbed rather than scored separately.
  3. For the four raw tissues (NETC/SNFH/ET/RC), lesions touching a common lesion of
     another tissue are merged cross-tissue (combine_lesions_tissues).
  4. The GT lesion floor is > 20 voxels (> 10 for ET), not >= 50.
  5. A tissue absent from both GT and prediction scores 1.0, not nan.

The official tool needs ~3-5 min/case (it round-trips every mask through NIfTI files and
computes HD95/NSD); this port computes Dice-only from arrays with bbox-cropped dilations
and runs in ~1 s/case, which is what makes a 1468-case threshold sweep feasible.

FAITHFULNESS NOTES (deliberate bug-compatibility — do not "fix" these):
  * SNFH gets dil_factor 3, not 5: the official code tests ``label_value == "SFNH"`` (a
    typo) everywhere it special-cases the factor.
  * Cross-tissue relabelling only happens when MORE THAN ONE (lesion, partner) touching
    pair exists (``len(touching_labels) > 1`` in combine_lesions_*), and a lesion touching
    two partner lesions joins the group of the first pair in sorted order only.
  * Dilation uses ``generate_binary_structure(3, 2)`` (18-connectivity); connected
    components use full 26-connectivity, matching ``cc3d.connected_components(...,
    connectivity=26)``.
  * "Touching" in the cross-tissue pass means overlap after a ONE-iteration dilation.

Verified against the official tool on 12 OOF cases x {raw, post-processed}: see the
cross-check artefacts from 2026-08-17. HD95/NSD are not ported.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, find_objects, generate_binary_structure
from scipy.ndimage import label as cc_label

TISSUES = {"WT": [1, 2, 3], "TC": [1, 3], "NETC": [1], "SNFH": [2], "ET": [3], "RC": [4]}
# combine_lesions_tissues order: each tissue is merged against these partners, chained.
CROSS_PARTNERS = {"NETC": ["ET", "RC"], "ET": ["NETC", "RC"],
                  "RC": ["ET", "NETC"], "SNFH": ["ET", "RC", "NETC"]}
_DIL_STRUCT = generate_binary_structure(3, 2)   # 18-conn: the official dilation struct
_CC_STRUCT = generate_binary_structure(3, 3)    # 26-conn: the official cc connectivity


def _dil_factor(tissue: str) -> int:
    # Official logic incl. the "SFNH" typo: only NETC and RC ever get 5.
    return 5 if tissue in ("NETC", "RC") else 3


def _volume_floor(tissue: str) -> int:
    return 10 if tissue == "ET" else 20


def _padded_slices(slices, shape, pad):
    return tuple(slice(max(0, s.start - pad), min(dim, s.stop + pad))
                 for s, dim in zip(slices, shape))


def _combined_cc(mask: np.ndarray, dil: int) -> np.ndarray:
    """Within-tissue dilation-merged connected components.

    Official recipe: cc26 the mask, dilate the mask by ``dil`` (18-conn struct), cc26 the
    dilated mask, and give every original component the id of the dilated component it
    falls in. Restricting the dilated-cc map to the mask is exactly that, in one pass.
    Dilation is computed inside the mask bbox padded by ``dil`` — binary dilation cannot
    grow further than one struct-step per iteration, so this is exact.
    """
    out = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return out
    box = _padded_slices(find_objects(mask.astype(np.int8), max_label=1)[0], mask.shape,
                         dil + 1)
    sub = mask[box]
    dilated = binary_dilation(sub, structure=_DIL_STRUCT, iterations=dil)
    lab, _ = cc_label(dilated, structure=_CC_STRUCT)
    out[box] = np.where(sub, lab, 0)
    return out


def _prune(labmap: np.ndarray, max_voxels: int) -> np.ndarray:
    """Delete lesions of <= max_voxels voxels (official pred-side pruning)."""
    if not labmap.any():
        return labmap
    sizes = np.bincount(labmap.ravel())
    sizes[0] = 0
    doomed = np.flatnonzero((sizes > 0) & (sizes <= max_voxels))
    if doomed.size:
        labmap = np.where(np.isin(labmap, doomed), 0, labmap)
    return labmap


def _renumber(labmap: np.ndarray) -> np.ndarray:
    """Relabel to consecutive 1..n (the official reorder_labels_nifti)."""
    ids = np.unique(labmap)
    ids = ids[ids != 0]
    if not ids.size:
        return labmap
    lut = np.zeros(int(labmap.max()) + 1, dtype=np.int32)
    lut[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    return lut[labmap]


def _touching_pairs(a_map: np.ndarray, b_map: np.ndarray) -> List[Tuple[int, int]]:
    """(a_label, b_label) pairs whose masks overlap after a 1-iteration 18-conn dilation.

    Overlap-after-dilation is symmetric in the two masks, so one direction reproduces the
    official get_touching_labels (which checks both)."""
    pairs = set()
    if not a_map.any() or not b_map.any():
        return []
    for a_id, sl in enumerate(find_objects(a_map), start=1):
        if sl is None:
            continue
        box = _padded_slices(sl, a_map.shape, 2)
        a_dil = binary_dilation(a_map[box] == a_id, structure=_DIL_STRUCT)
        hits = np.unique(b_map[box][a_dil])
        for b_id in hits[hits != 0]:
            pairs.add((a_id, int(b_id)))
    return sorted(pairs)


def _relabel_by_pairs(labmap: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    """The official relabel_nifti_image: only fires when >1 pair exists; groups this
    tissue's labels by the partner label of their FIRST pair in sorted order."""
    if len(pairs) <= 1:
        return labmap
    nxt = int(labmap.max()) + 1
    group_of_partner: Dict[int, int] = {}
    for _, b in sorted(pairs):
        if b not in group_of_partner:
            group_of_partner[b] = nxt
            nxt += 1
    mapping: Dict[int, int] = {}
    for a, b in sorted(pairs):
        if a not in mapping:
            mapping[a] = group_of_partner[b]
    lut = np.arange(int(labmap.max()) + 1, dtype=np.int32)
    for a, new in mapping.items():
        lut[a] = new
    return lut[labmap]


def _cross_combine(maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """combine_lesions_tissues: chain-merge each raw tissue against its partners.

    Partners are always the tissue's ORIGINAL within-merged map (the official code reads
    the *_cc files, not the combined outputs), while the tissue itself accumulates
    relabels along its chain."""
    out = {}
    for tissue, partners in CROSS_PARTNERS.items():
        cur = maps[tissue]
        for p in partners:
            cur = _relabel_by_pairs(cur, _touching_pairs(cur, maps[p]))
        out[tissue] = _renumber(cur)
    return out


def _prepare(seg: np.ndarray, prune: bool, return_base: bool = False):
    """Full official label-map preparation for one segmentation.

    Returns per-tissue lesion label maps: within-tissue dilation-merged, pred-side pruned
    (prune=True), cross-tissue combined for the four raw tissues, renumbered. With
    ``return_base`` also returns the pre-cross-combine maps, which threshold sweeps reuse
    as partner maps — this is THE preparation pipeline; do not re-derive it elsewhere, or
    a faithfulness fix here silently forks the metric."""
    maps = {}
    for tissue, labels in TISSUES.items():
        m = _combined_cc(np.isin(seg, labels), _dil_factor(tissue))
        if prune:
            m = _prune(m, _volume_floor(tissue))
        maps[tissue] = m
    combined = _cross_combine(maps)
    final = {t: _renumber(combined.get(t, maps[t])) for t in TISSUES}
    return (maps, final) if return_base else final


def prepare_pred(seg: np.ndarray, return_base: bool = False):
    """Official pred-side preparation (pruned). See _prepare for return_base."""
    return _prepare(seg.astype(np.int32), prune=True, return_base=return_base)


def _score_tissue(pred_map: np.ndarray, gt_map: np.ndarray, tissue: str) -> Dict[str, float]:
    """Official per-tissue lesion-wise Dice from prepared lesion label maps."""
    dil = _dil_factor(tissue)
    floor = _volume_floor(tissue)
    n_gt = int(gt_map.max())
    n_pred = int(pred_map.max())

    pred_sizes = np.bincount(pred_map.ravel(), minlength=n_pred + 1)

    matched: set = set()
    kept_dices: List[float] = []
    n_fn = 0
    for gid, sl in enumerate(find_objects(gt_map, max_label=n_gt), start=1):
        if sl is None:
            continue
        box = _padded_slices(sl, gt_map.shape, dil + 1)
        gt_lesion = gt_map[box] == gid
        gt_vol = int(gt_lesion.sum())
        search = binary_dilation(gt_lesion, structure=_DIL_STRUCT, iterations=dil)
        hits = np.unique(pred_map[box][search])
        hits = hits[hits != 0]
        matched.update(int(h) for h in hits)

        if gt_vol <= floor:
            continue                    # row excluded from the mean, but hits still
                                        # count as used (they are not FPs)
        if hits.size == 0:
            kept_dices.append(0.0)
            n_fn += 1
            continue
        # Dice of the UNION of intersecting pred lesions vs this (undilated) GT lesion.
        # The union can extend beyond this bbox, so take sizes from the global table and
        # the intersection from the bbox (a superset of the GT lesion).
        pred_in_box = np.isin(pred_map[box], hits)
        inter = int(np.logical_and(pred_in_box, gt_lesion).sum())
        denom = int(pred_sizes[hits].sum()) + gt_vol
        kept_dices.append(2.0 * inter / denom if denom else 0.0)

    n_fp = sum(1 for pid in range(1, n_pred + 1)
               if pid not in matched and pred_sizes[pid] > 0)

    denom = len(kept_dices) + n_fp
    if denom == 0:
        # Official convention: nothing scoreable (incl. absent-in-both) => 1.0
        return {"dice": 1.0, "n_scored_gt": 0, "n_fp": 0, "n_fn": 0, "absent": True}
    return {"dice": float(np.sum(kept_dices) / denom),
            "n_scored_gt": len(kept_dices), "n_fp": n_fp, "n_fn": n_fn, "absent": False}


def official_case(pred: np.ndarray, gt: np.ndarray,
                  tissues: Sequence[str] = tuple(TISSUES),
                  gt_maps: Dict[str, np.ndarray] | None = None) -> Dict[str, Dict]:
    """Official lesion-wise Dice for every requested tissue of one case.

    ``gt_maps``: pass the result of ``prepare_gt(gt)`` to amortise GT preparation across
    many pred variants of the same case (threshold sweeps)."""
    if gt_maps is None:
        gt_maps = prepare_gt(gt)
    pred_maps = prepare_pred(pred)
    return {t: _score_tissue(pred_maps[t], gt_maps[t], t) for t in tissues}


def prepare_gt(gt: np.ndarray) -> Dict[str, np.ndarray]:
    return _prepare(gt.astype(np.int32), prune=False)
