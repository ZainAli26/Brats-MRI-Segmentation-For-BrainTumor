"""Heuristic BraTS post-processing (fixed thresholds) — NOT what nnU-Net does.

Three techniques used by top BraTS solutions:
  1. Connected Component Analysis  — clean up small components per class
  2. ET Volume Suppression         — convert tiny ET clusters to the tumor-core class
  3. Morphological Hole Filling    — fill interior background holes in WT/TC

These thresholds are hand-picked, not validated on this dataset. nnU-Net instead *derives*
its post-processing from the out-of-fold predictions and only ever removes all-but-the-largest
component — see ``src/evaluation/nnunet_postprocessing.py``. Use this module when explicitly
comparing against the BraTS-competition recipe; use that one to match nnU-Net.

Class-count agnostic: the foreground labels, the ET label, and the WT/TC label
sets are all parameters, so the same routine serves the BraTS-2024 5-class scheme
(1=NETC, 2=SNFH, 3=ET, 4=RC) and the BraTS-2023 4-class scheme (1=NCR, 2=ED, 3=ET).
Defaults match BraTS-2024.

Two defaults differ from the original implementation because the original was measurably
wrong; pass the legacy values to reproduce older numbers:

* ``small_component_mode="reassign"`` (was: always zero). Zeroing a sub-threshold component
  that sits *inside* the tumor punches a hole through WT and TC, costing Dice on regions that
  were otherwise correct. Only components with no foreground neighbour are true noise and get
  zeroed; interior speckle is relabelled to its dominant neighbouring foreground class, which
  leaves WT/TC intact. Matters most for BraTS-2024, where NETC and RC are a tiny voxel
  fraction and a 50-voxel floor otherwise deletes genuine structure. Legacy: ``"always"``.
* ``fill_holes_axial=False`` (was: True). Filling per axial slice closes concavities that are
  open in z and are not holes in 3D, inventing voxels. Legacy: ``True``.

Component labelling is 26-neighbour, matching nnU-Net/skimage rather than scipy's 6-neighbour
default. Hole filling keeps a 6-connected background flood fill, the conventional dual — see
``_fill_holes``.
"""

import numpy as np
import torch
from scipy.ndimage import (
    binary_dilation, binary_fill_holes, find_objects, generate_binary_structure,
)
from scipy.ndimage import label as scipy_label

# 26-connectivity in 3D — what nnU-Net (via skimage's connectivity=ndim) uses.
_FULL_3D = generate_binary_structure(3, 3)


def _structure(ndim: int) -> np.ndarray:
    return _FULL_3D if ndim == 3 else generate_binary_structure(ndim, ndim)


# ── helpers ──────────────────────────────────────────────────────────────────

def _pad_slices(slices, shape, margin: int = 1):
    """Grow a ``find_objects`` bounding box by ``margin``, clipped to ``shape``."""
    return tuple(slice(max(0, s.start - margin), min(dim, s.stop + margin))
                 for s, dim in zip(slices, shape))


def _clean_small_components(post: np.ndarray, cls: int, min_size: int, mode: str) -> None:
    """In-place cleanup of sub-``min_size`` components of ``cls``.

    ``mode="always"`` zeroes every small component (legacy). ``mode="reassign"`` zeroes only
    components with no foreground neighbour and relabels the rest to their dominant
    neighbouring foreground class, so interior speckle does not become a hole in WT/TC.
    """
    struct = _structure(post.ndim)
    cls_mask = post == cls
    if not cls_mask.any():
        return
    labeled, n = scipy_label(cls_mask, structure=struct)
    if n == 0:
        return
    boxes = find_objects(labeled)
    for cid in range(1, n + 1):
        box = boxes[cid - 1]
        if box is None:
            continue
        # Work inside a 1-voxel-padded bounding box; neighbours of the component all live there.
        sub = _pad_slices(box, post.shape)
        comp_local = labeled[sub] == cid
        if comp_local.sum() >= min_size:
            continue
        if mode == "always":
            post[sub][comp_local] = 0
            continue
        ring = binary_dilation(comp_local, structure=struct) & ~comp_local
        neighbours = post[sub][ring]
        neighbours = neighbours[(neighbours != 0) & (neighbours != cls)]
        if neighbours.size == 0:
            post[sub][comp_local] = 0   # isolated island -> genuine noise
        else:
            post[sub][comp_local] = np.bincount(neighbours).argmax()


def _fill_holes(mask: np.ndarray, axial: bool = False) -> np.ndarray:
    """Fill interior holes in a 3-D binary mask.

    True 3-D fill by default. ``axial=True`` reproduces the legacy slice-by-slice behaviour,
    which also closes concavities that are open along z and therefore adds voxels that are
    not enclosed in 3D.

    Left at scipy's default 6-connected background flood fill: that is the conventional dual
    of a 26-connected foreground, and a 26-connected background would let it escape through
    diagonal gaps and fill almost nothing.
    """
    if not axial:
        return binary_fill_holes(mask)
    filled = np.zeros_like(mask)
    for z in range(mask.shape[2]):
        filled[:, :, z] = binary_fill_holes(mask[:, :, z])
    return filled


# ── main API ─────────────────────────────────────────────────────────────────

def postprocess_prediction(
    pred: np.ndarray,
    et_min_voxels: int = 250,
    min_component_size: int = 50,
    fill_holes: bool = True,
    foreground_classes=(1, 2, 3, 4),
    et_label: int = 3,
    et_fold_into: int = 1,
    wt_labels=(1, 2, 3),
    tc_labels=(1, 3),
    fill_into: int = 1,
    small_component_mode: str = "reassign",
    fill_holes_axial: bool = False,
) -> np.ndarray:
    """Apply standard BraTS post-processing to an argmax prediction volume.

    Args:
        pred: (H, W, D) integer array of class labels.
        et_min_voxels: If total predicted ET volume is below this threshold,
              fold all ET voxels into ``et_fold_into`` (keeps them inside TC).
              BraTS challenge winners commonly use 250–500 voxels.
        min_component_size: Clean up connected components with fewer than this many
              voxels in each foreground class. See ``small_component_mode``. Set to 0
              or 1 to skip this step.
        fill_holes: Fill interior background holes in WT and TC masks.
        foreground_classes: Labels to run connected-component filtering on.
              Defaults to the 4 BraTS-2024 foreground labels.
        et_label: Label index of enhancing tissue (3 in both 2023 and 2024).
        et_fold_into: Label that suppressed ET is folded into (tumor-core class).
        wt_labels / tc_labels: Label sets composing WT / TC for hole filling.
        fill_into: Label assigned to newly filled hole voxels.
        small_component_mode: ``"reassign"`` (default) zeroes only components with no
              foreground neighbour and relabels interior speckle to its dominant
              neighbouring foreground class, preserving WT/TC. ``"always"`` zeroes every
              small component (legacy behaviour; punches holes through WT/TC).
        fill_holes_axial: ``True`` restores the legacy per-axial-slice fill instead of a
              true 3-D fill.

    Returns:
        Post-processed prediction array of same shape and dtype as `pred`.
    """
    if small_component_mode not in ("reassign", "always"):
        raise ValueError(f"small_component_mode must be 'reassign' or 'always', "
                         f"got {small_component_mode!r}")
    post = pred.copy()

    # Step 1 — connected component cleanup per class
    if min_component_size > 1:
        for cls in foreground_classes:
            _clean_small_components(post, int(cls), min_component_size, small_component_mode)

    # Step 2 — ET suppression: tiny ET is almost always a false positive
    et_mask = post == et_label
    if 0 < et_mask.sum() < et_min_voxels:
        post[et_mask] = et_fold_into  # Fold into tumor-core class so it stays inside TC

    # Step 3 — morphological hole filling for WT and TC
    if fill_holes:
        # Whole Tumour
        wt_mask = np.isin(post, list(wt_labels))
        wt_filled = _fill_holes(wt_mask, fill_holes_axial)
        # Voxels added by hole filling belong to the dominant surrounding class.
        # We assign them to `fill_into` — safest anatomical assumption.
        new_wt = wt_filled & ~wt_mask
        post[new_wt] = fill_into

        # Tumour Core
        tc_mask = np.isin(post, list(tc_labels))
        tc_filled = _fill_holes(tc_mask, fill_holes_axial)
        new_tc = tc_filled & ~tc_mask
        # Only fill inside WT (don't extend beyond tumour boundary)
        new_tc &= wt_filled
        post[new_tc] = fill_into

    return post.astype(pred.dtype)


def postprocess_kwargs_from_config(config: dict) -> dict:
    """Derive the label-dependent post-processing kwargs from an experiment config.

    Pulls the ET label and the WT/TC label sets from ``evaluation.regions`` and the
    full foreground set from ``data.num_classes`` so post-processing matches whatever
    class scheme the run used.
    """
    regions = config.get("evaluation", {}).get("regions", {})
    num_classes = int(config.get("data", {}).get("num_classes", 4))
    foreground = tuple(range(1, num_classes))
    et_region = regions.get("ET", [3])
    et_label = et_region[0] if et_region else 3
    wt_labels = tuple(regions.get("WT", foreground))
    tc_labels = tuple(regions.get("TC", (1, et_label)))
    # Fold suppressed ET into the non-enhancing core (smallest TC label that isn't ET).
    core_candidates = [c for c in tc_labels if c != et_label]
    et_fold_into = core_candidates[0] if core_candidates else 1
    return dict(
        foreground_classes=foreground,
        et_label=et_label,
        et_fold_into=et_fold_into,
        wt_labels=wt_labels,
        tc_labels=tc_labels,
        fill_into=et_fold_into,
    )


def postprocess_batch_tensor(
    pred_tensor: torch.Tensor,
    et_min_voxels: int = 250,
    min_component_size: int = 50,
    fill_holes: bool = True,
    **label_kwargs,
) -> torch.Tensor:
    """Convenience wrapper: takes (B, H, W, D) argmax tensor, returns same shape."""
    device = pred_tensor.device
    pred_np = pred_tensor.cpu().numpy()
    out = np.stack([
        postprocess_prediction(pred_np[i], et_min_voxels, min_component_size,
                               fill_holes, **label_kwargs)
        for i in range(pred_np.shape[0])
    ])
    return torch.from_numpy(out).to(device)
