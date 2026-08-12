"""nnU-Net v2 sliding-window inference: Gaussian tiling + mirroring TTA.

Ported rather than delegated to MONAI's ``sliding_window_inference`` because three
details differ and each costs Dice:

* **Step placement.** nnU-Net computes ``ceil((size - patch) / (patch * 0.5)) + 1`` steps
  and then spreads them *evenly* over the axis; MONAI walks a fixed stride and clamps the
  last window. Different tiles ⇒ different border predictions.
* **Gaussian map.** A dirac blurred with ``sigma = patch/8``, normalised to max 1, with
  exact zeros lifted to the smallest nonzero value so the weight map can never divide by 0.
* **Aggregation.** Raw logits are accumulated (weighted), divided by the accumulated
  weight, and only *then* softmaxed. Mirroring TTA likewise averages **logits** across the
  eight flips, not per-flip softmax outputs.

``predict_case`` additionally undoes preprocessing — un-crop back into the original
uncropped volume — so predictions can be scored against the untouched ground truth.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


def compute_gaussian(tile_size: Sequence[int], sigma_scale: float = 1.0 / 8,
                     dtype=torch.float32, device=torch.device("cpu")) -> torch.Tensor:
    from scipy.ndimage import gaussian_filter

    tmp = np.zeros(tuple(tile_size))
    tmp[tuple(i // 2 for i in tile_size)] = 1
    g = gaussian_filter(tmp, [i * sigma_scale for i in tile_size], 0, mode="constant", cval=0)
    g = g / np.max(g)
    g = g.astype(np.float32)
    g[g == 0] = np.min(g[g != 0])   # never zero — it becomes a divisor
    return torch.from_numpy(g).to(dtype=dtype, device=device)


def compute_steps_for_sliding_window(image_size: Sequence[int], tile_size: Sequence[int],
                                     tile_step_size: float = 0.5) -> List[List[int]]:
    assert all(i >= j for i, j in zip(image_size, tile_size)), \
        f"image {list(image_size)} smaller than patch {list(tile_size)} — pad first"
    target_steps = [i * tile_step_size for i in tile_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_steps, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        max_step = image_size[dim] - tile_size[dim]
        actual = max_step / (num_steps[dim] - 1) if num_steps[dim] > 1 else 1e12
        steps.append([int(np.round(actual * i)) for i in range(num_steps[dim])])
    return steps


def _mirror_axis_sets(axes: Sequence[int]) -> List[Tuple[int, ...]]:
    sets: List[Tuple[int, ...]] = [()]
    for r in range(1, len(axes) + 1):
        sets.extend(combinations(axes, r))
    return sets


def _maybe_mirror_and_predict(network: nn.Module, x: torch.Tensor,
                              mirror_axes: Optional[Sequence[int]]) -> torch.Tensor:
    """Average the network's LOGITS over every flip combination of ``mirror_axes``."""
    out = network(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    if not mirror_axes:
        return out

    spatial = [a + 2 for a in mirror_axes]
    n = 1
    for flips in _mirror_axis_sets(list(range(len(mirror_axes)))):
        if not flips:
            continue
        dims = tuple(spatial[i] for i in flips)
        pred = network(torch.flip(x, dims))
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        out = out + torch.flip(pred, dims)
        n += 1
    return out / n


def _pad_to_patch(data: torch.Tensor, patch_size: Sequence[int]) -> Tuple[torch.Tensor, List[int]]:
    """Symmetric zero-pad so every axis is at least ``patch_size``; returns the padding."""
    shape = data.shape[1:]
    pad: List[int] = []
    for i in range(len(shape) - 1, -1, -1):   # F.pad wants last-axis-first
        need = max(0, patch_size[i] - shape[i])
        pad.extend([need // 2, need - need // 2])
    if any(pad):
        data = torch.nn.functional.pad(data[None], pad, mode="constant", value=0)[0]
    return data, pad


@torch.inference_mode()
def predict_sliding_window(
    network: nn.Module,
    data: torch.Tensor,
    patch_size: Sequence[int],
    num_classes: int,
    tile_step_size: float = 0.5,
    use_gaussian: bool = True,
    mirror_axes: Optional[Sequence[int]] = (0, 1, 2),
    device: torch.device = torch.device("cuda"),
    amp: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """Return per-voxel logits (C, X, Y, Z) for one preprocessed case."""
    assert data.ndim == 4, "data must be (C, X, Y, Z)"
    network.eval()

    padded, pad = _pad_to_patch(data, patch_size)
    padded_shape = padded.shape[1:]

    # Accumulate on the GPU when it fits, otherwise on the CPU (a full BraTS volume of
    # 5 float32 classes is ~65 MB, so this is only a safety valve for larger label sets).
    acc_device = device
    logits = torch.zeros((num_classes, *padded_shape), dtype=torch.float32, device=acc_device)
    weights = torch.zeros(padded_shape, dtype=torch.float32, device=acc_device)

    gaussian = (compute_gaussian(patch_size, device=acc_device) if use_gaussian
                else torch.ones(tuple(patch_size), device=acc_device))

    steps = compute_steps_for_sliding_window(padded_shape, patch_size, tile_step_size)
    if verbose:
        print(f"sliding window: {np.prod([len(s) for s in steps])} tiles over {list(padded_shape)}")

    autocast = torch.autocast(device.type, enabled=amp) if device.type == "cuda" \
        else torch.autocast("cpu", enabled=False)

    for sx in steps[0]:
        for sy in steps[1]:
            for sz in steps[2]:
                sl = tuple(slice(s, s + p) for s, p in zip((sx, sy, sz), patch_size))
                tile = padded[(slice(None), *sl)][None].to(device, non_blocking=True)
                with autocast:
                    pred = _maybe_mirror_and_predict(network, tile, mirror_axes)
                pred = pred[0].to(acc_device, dtype=torch.float32)
                logits[(slice(None), *sl)] += pred * gaussian
                weights[sl] += gaussian

    logits /= weights[None]

    if any(pad):
        # pad was built last-axis-first; walk it back to slice off the padding.
        slicer = [slice(None)]
        for i in range(len(padded_shape)):
            lo = pad[2 * (len(padded_shape) - 1 - i)]
            hi = pad[2 * (len(padded_shape) - 1 - i) + 1]
            slicer.append(slice(lo, padded_shape[i] - hi))
        logits = logits[tuple(slicer)]
    return logits


def logits_to_segmentation(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, 0).argmax(0).cpu().numpy().astype(np.uint8)


def undo_cropping(seg_cropped: np.ndarray, properties: Dict) -> np.ndarray:
    """Place a prediction back into the original (uncropped) volume, zeros elsewhere.

    nnU-Net's ``convert_predicted_logits_to_segmentation_with_correct_shape`` resamples the
    *logits* back to the original spacing before the argmax, then un-crops. This port skips
    the resampling step because BraTS is already at the plans' 1 mm isotropic target, making
    it a no-op. Any plan that actually resamples needs that step added — checked explicitly
    here rather than left to fail as a confusing broadcast error.
    """
    pre = properties.get("shape_after_cropping_and_before_resampling")
    post = properties.get("shape_after_resampling")
    if pre is not None and post is not None and list(pre) != list(post):
        raise NotImplementedError(
            f"Preprocessing resampled this case ({list(pre)} -> {list(post)}), but inference "
            f"argmaxes at the resampled resolution and only undoes cropping. Resample the "
            f"logits back to {list(pre)} before argmax (nnU-Net does this) — otherwise the "
            f"prediction does not align with the ground truth."
        )
    full = np.zeros(properties["shape_before_cropping"], dtype=seg_cropped.dtype)
    bbox = properties["bbox_used_for_cropping"]
    slicer = tuple(slice(lo, hi) for lo, hi in bbox)
    full[slicer] = seg_cropped
    return full


def predict_case(
    network: nn.Module,
    data: np.ndarray,
    properties: Dict,
    patch_size: Sequence[int],
    num_classes: int,
    device: torch.device = torch.device("cuda"),
    tile_step_size: float = 0.5,
    use_gaussian: bool = True,
    mirror_axes: Optional[Sequence[int]] = (0, 1, 2),
    amp: bool = True,
    return_to_original_space: bool = True,
) -> np.ndarray:
    """Predict one preprocessed case and (by default) map it back to the raw volume."""
    tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
    logits = predict_sliding_window(
        network, tensor, patch_size, num_classes, tile_step_size=tile_step_size,
        use_gaussian=use_gaussian, mirror_axes=mirror_axes, device=device, amp=amp,
    )
    seg = logits_to_segmentation(logits)
    return undo_cropping(seg, properties) if return_to_original_space else seg
