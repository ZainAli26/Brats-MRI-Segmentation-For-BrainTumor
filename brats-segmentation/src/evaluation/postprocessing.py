"""Post-processing for BraTS segmentation predictions.

Three standard techniques used by top BraTS solutions:
  1. Connected Component Analysis  — drop small isolated components per class
  2. ET Volume Suppression         — convert tiny ET clusters to NCR (class 1)
  3. Morphological Hole Filling    — fill interior background holes in WT/TC
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
) -> np.ndarray:
    """Apply standard BraTS post-processing to an argmax prediction volume.

    Args:
        pred: (H, W, D) int32 array with values {0, 1, 2, 3}
              0=background, 1=NCR, 2=ED, 3=ET
        et_min_voxels: If total predicted ET volume is below this threshold,
              convert all ET voxels to NCR (class 1). BraTS challenge winners
              commonly use 250–500 voxels.
        min_component_size: Remove connected components with fewer than this
              many voxels from each foreground class. Prunes isolated noise.
        fill_holes: Fill interior background holes in WT and TC masks.

    Returns:
        Post-processed prediction array of same shape and dtype as `pred`.
    """
    post = pred.copy()

    # Step 1 — connected component filtering per class
    for cls in [1, 2, 3]:  # NCR, ED, ET
        cls_mask = (post == cls).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue
        filtered = _remove_small_components(cls_mask.astype(bool), min_component_size)
        removed = cls_mask.astype(bool) & ~filtered
        post[removed] = 0

    # Step 2 — ET suppression: tiny ET is almost always a false positive
    et_mask = post == 3
    if 0 < et_mask.sum() < et_min_voxels:
        post[et_mask] = 1  # Fold into NCR so it stays inside TC

    # Step 3 — morphological hole filling for WT and TC
    if fill_holes:
        # Whole Tumour = NCR + ED + ET (classes 1, 2, 3)
        wt_mask = (post > 0)
        wt_filled = _fill_holes_3d(wt_mask)
        # Voxels added by hole filling belong to the dominant surrounding class.
        # We assign them as NCR (1) — safest anatomical assumption.
        new_wt = wt_filled & ~wt_mask
        post[new_wt] = 1

        # Tumour Core = NCR + ET (classes 1, 3)
        tc_mask = (post == 1) | (post == 3)
        tc_filled = _fill_holes_3d(tc_mask)
        new_tc = tc_filled & ~tc_mask
        # Only fill inside WT (don't extend beyond tumour boundary)
        new_tc &= wt_filled
        post[new_tc] = 1

    return post.astype(pred.dtype)


def postprocess_batch_tensor(
    pred_tensor: torch.Tensor,
    et_min_voxels: int = 250,
    min_component_size: int = 50,
    fill_holes: bool = True,
) -> torch.Tensor:
    """Convenience wrapper: takes (B, H, W, D) argmax tensor, returns same shape."""
    device = pred_tensor.device
    pred_np = pred_tensor.cpu().numpy()
    out = np.stack([
        postprocess_prediction(pred_np[i], et_min_voxels, min_component_size, fill_holes)
        for i in range(pred_np.shape[0])
    ])
    return torch.from_numpy(out).to(device)
