"""nnU-Net v2-style inference: Gaussian-weighted sliding window + mirroring TTA.

The default MONAI inference used elsewhere (constant window weighting, no test-time
augmentation) under-scores vs nnU-Net. This module reproduces nnU-Net's inference:

  * sliding window with Gaussian importance weighting (center voxels weighted higher)
  * test-time augmentation by mirroring over spatial axes (softmax averaged)

It is architecture-agnostic — it only calls `model(x)` — so it keeps working if you
swap the network later. Returns probabilities so callers can argmax + post-process.
"""

from itertools import combinations
from typing import List, Sequence

import torch
from monai.inferers import sliding_window_inference

from src.utils import inference_wrapper


def _mirror_axis_sets(axes: Sequence[int]) -> List[tuple]:
    """All flip combinations over `axes` (incl. the empty = no-flip set).

    nnU-Net mirrors over every spatial axis -> 2**len(axes) forward passes.
    """
    sets = [()]
    for r in range(1, len(axes) + 1):
        sets.extend(combinations(axes, r))
    return sets


def predict_probabilities(
    model,
    image: torch.Tensor,
    roi_size: Sequence[int],
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    mode: str = "gaussian",
    tta: bool = False,
    tta_axes: Sequence[int] = (0, 1, 2),
) -> torch.Tensor:
    """Run sliding-window inference and return softmax probabilities.

    Args:
        image: (B, C, H, W, D) input.
        roi_size: sliding-window patch size (the training patch / spatial_size).
        mode: "gaussian" (nnU-Net) or "constant".
        tta: if True, average softmax over axis-mirroring flips (nnU-Net TTA).
        tta_axes: spatial axes to mirror (0,1,2 -> X,Y,Z), as offsets from the
                  spatial dims (channel/batch handled internally).

    Returns:
        (B, C, H, W, D) probability tensor (softmax over channels).
    """
    fn = inference_wrapper(model)

    def _sw(x):
        return sliding_window_inference(
            x, roi_size, sw_batch_size, fn, overlap=overlap, mode=mode
        )

    if not tta:
        return torch.softmax(_sw(image), dim=1)

    # Spatial axes in the (B, C, H, W, D) tensor are dims 2,3,4.
    spatial_dims = [a + 2 for a in tta_axes]
    prob_sum = None
    n = 0
    for flip_set in _mirror_axis_sets(list(range(len(tta_axes)))):
        dims = tuple(spatial_dims[i] for i in flip_set)
        x = torch.flip(image, dims=dims) if dims else image
        logits = _sw(x)
        probs = torch.softmax(logits, dim=1)
        if dims:  # undo the flip so predictions realign with the input
            probs = torch.flip(probs, dims=dims)
        prob_sum = probs if prob_sum is None else prob_sum + probs
        n += 1
    return prob_sum / n


def inference_kwargs_from_config(config: dict) -> dict:
    """Read the optional ``inference:`` config block (defaults = old behavior).

    Absent block -> constant weighting, no TTA (unchanged for exp01–18).
    """
    inf = config.get("inference", {}) or {}
    return dict(
        mode=inf.get("mode", "constant"),
        tta=inf.get("tta", False),
        overlap=inf.get("overlap", config.get("training", {}).get("sw_overlap", 0.5)),
        postprocess=inf.get("postprocess", False),
    )
