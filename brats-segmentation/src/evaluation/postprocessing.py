"""Post-processing for BraTS segmentation predictions.

Three standard techniques used by top BraTS solutions:
  1. Connected Component Analysis  — drop small isolated components per class
  2. ET Volume Suppression         — convert tiny ET clusters to the tumor-core class
  3. Morphological Hole Filling    — fill interior background holes in WT/TC

Class-count agnostic: the foreground labels, the ET label, and the WT/TC label
sets are all parameters, so the same routine serves the BraTS-2024 5-class scheme
(1=NETC, 2=SNFH, 3=ET, 4=RC) and the BraTS-2023 4-class scheme (1=NCR, 2=ED, 3=ET).
Defaults match BraTS-2024.
"""

import numpy as np
import torch
from scipy.ndimage import label as scipy_label, binary_fill_holes


# ── helpers ──────────────────────────────────────────────────────────────────

def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Return binary mask with connected components < min_size voxels removed."""
    labeled, n = scipy_label(mask)
    if n == 0:
        return mask
    out = np.zeros_like(mask)
    for cid in range(1, n + 1):
        component = labeled == cid
        if component.sum() >= min_size:
            out |= component
    return out


def _fill_holes_3d(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a 3-D binary mask slice-by-slice (axial plane)."""
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
) -> np.ndarray:
    """Apply standard BraTS post-processing to an argmax prediction volume.

    Args:
        pred: (H, W, D) integer array of class labels.
        et_min_voxels: If total predicted ET volume is below this threshold,
              fold all ET voxels into ``et_fold_into`` (keeps them inside TC).
              BraTS challenge winners commonly use 250–500 voxels.
        min_component_size: Remove connected components with fewer than this
              many voxels from each foreground class. Prunes isolated noise.
        fill_holes: Fill interior background holes in WT and TC masks.
        foreground_classes: Labels to run connected-component filtering on.
              Defaults to the 4 BraTS-2024 foreground labels.
        et_label: Label index of enhancing tissue (3 in both 2023 and 2024).
        et_fold_into: Label that suppressed ET is folded into (tumor-core class).
        wt_labels / tc_labels: Label sets composing WT / TC for hole filling.
        fill_into: Label assigned to newly filled hole voxels.

    Returns:
        Post-processed prediction array of same shape and dtype as `pred`.
    """
    post = pred.copy()

    # Step 1 — connected component filtering per class
    for cls in foreground_classes:
        cls_mask = (post == cls).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue
        filtered = _remove_small_components(cls_mask.astype(bool), min_component_size)
        removed = cls_mask.astype(bool) & ~filtered
        post[removed] = 0

    # Step 2 — ET suppression: tiny ET is almost always a false positive
    et_mask = post == et_label
    if 0 < et_mask.sum() < et_min_voxels:
        post[et_mask] = et_fold_into  # Fold into tumor-core class so it stays inside TC

    # Step 3 — morphological hole filling for WT and TC
    if fill_holes:
        # Whole Tumour
        wt_mask = np.isin(post, list(wt_labels))
        wt_filled = _fill_holes_3d(wt_mask)
        # Voxels added by hole filling belong to the dominant surrounding class.
        # We assign them to `fill_into` — safest anatomical assumption.
        new_wt = wt_filled & ~wt_mask
        post[new_wt] = fill_into

        # Tumour Core
        tc_mask = np.isin(post, list(tc_labels))
        tc_filled = _fill_holes_3d(tc_mask)
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
