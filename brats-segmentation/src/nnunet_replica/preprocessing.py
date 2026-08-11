"""nnU-Net v2 preprocessing, reimplemented (crop → normalize → resample → cache).

This is a faithful port of ``DefaultPreprocessor.run_case`` plus
``crop_to_nonzero`` and ``ZScoreNormalization``. It exists so the custom loop eats
exactly what the native trainer eats:

  1. read all channels + seg with SimpleITK (nnU-Net's ``SimpleITKIO``, so the array
     axis order and hence the patch-size axis order match the plan);
  2. crop to the nonzero (brain) bounding box, writing ``-1`` into the seg outside it —
     that ``-1`` is what later marks "outside the brain" for the masked z-score and for
     ``MaskTransform`` during augmentation;
  3. per-channel z-score over the ``seg >= 0`` region only (``use_mask_for_norm``),
     leaving the outside at 0;
  4. resample to the plan's target spacing (a no-op for BraTS, which is already 1 mm —
     implemented anyway so a re-planned dataset does not silently skip it);
  5. sample up to 10 000 voxel locations per foreground class — this is what makes
     nnU-Net's 33 % forced-foreground patch sampling O(1) at train time.

The cache is written in nnU-Net's own *unpacked* layout (``<case>.npy`` /
``<case>_seg.npy`` / ``<case>.pkl``) so it can be memory-mapped per patch instead of
re-decompressing four ``.nii.gz`` every epoch.

One deliberate deviation: ``store_dtype`` defaults to float16 rather than nnU-Net's
float32. The images are z-scored (values ~ ±5) and training runs under AMP, so fp16
costs nothing measurable and halves an otherwise ~86 GB cache for 1621 BraTS cases.
Set ``store_dtype: float32`` to match nnU-Net byte for byte.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.nnunet_replica.plans import ConfigurationPlan


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------

def read_image_sitk(path: str) -> Tuple[np.ndarray, Dict]:
    """Read one image the way nnU-Net's ``SimpleITKIO`` does.

    Returns ``(array, properties)`` with the array in SimpleITK's (z, y, x) order and
    the spacing reversed to match — this is the axis convention the plan's patch size
    is expressed in, so it must not be "fixed" to nibabel's (x, y, z).
    """
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (z, y, x)
    props = {
        "sitk_spacing": img.GetSpacing(),          # (x, y, z)
        "sitk_origin": img.GetOrigin(),
        "sitk_direction": img.GetDirection(),
        "spacing": list(img.GetSpacing())[::-1],   # (z, y, x) — matches the array
    }
    return arr, props


def write_segmentation_like(reference_path: str, seg: np.ndarray, out_path: str) -> None:
    """Write a (z, y, x) label array as NIfTI, copying geometry from a reference image."""
    import SimpleITK as sitk

    ref = sitk.ReadImage(str(reference_path))
    out = sitk.GetImageFromArray(seg.astype(np.uint8))
    out.SetSpacing(ref.GetSpacing())
    out.SetOrigin(ref.GetOrigin())
    out.SetDirection(ref.GetDirection())
    sitk.WriteImage(out, str(out_path))


# --------------------------------------------------------------------------------------
# Cropping (port of nnunetv2.preprocessing.cropping.cropping)
# --------------------------------------------------------------------------------------

def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    """True wherever ANY channel is nonzero, with interior holes filled."""
    from scipy.ndimage import binary_fill_holes

    assert data.ndim == 4, "data must be (C, X, Y, Z)"
    mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        mask |= data[c] != 0
    return binary_fill_holes(mask)


def bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    """Tight [[lo, hi), ...] bounding box of a boolean mask, one pair per axis."""
    bbox = []
    for axis in range(mask.ndim):
        other = tuple(i for i in range(mask.ndim) if i != axis)
        idx = np.where(mask.any(axis=other))[0]
        bbox.append([int(idx[0]), int(idx[-1]) + 1])
    return bbox


def crop_to_nonzero(
    data: np.ndarray, seg: Optional[np.ndarray] = None, nonzero_label: int = -1
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Crop to the brain box; mark voxels outside the brain with ``nonzero_label``."""
    nonzero_mask = create_nonzero_mask(data)
    bbox = bbox_from_mask(nonzero_mask)
    slicer = tuple(slice(lo, hi) for lo, hi in bbox)

    data = data[(slice(None), *slicer)]
    nonzero_mask = nonzero_mask[slicer][None]

    if seg is not None:
        seg = seg[(slice(None), *slicer)]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, 0, nonzero_label).astype(np.int8)
    return data, seg, bbox


# --------------------------------------------------------------------------------------
# Normalization + resampling
# --------------------------------------------------------------------------------------

def zscore_normalize(image: np.ndarray, seg: np.ndarray, use_mask: bool) -> np.ndarray:
    """nnU-Net ``ZScoreNormalization``. With ``use_mask`` the outside stays exactly 0."""
    image = image.astype(np.float32, copy=True)
    if use_mask:
        mask = seg[0] >= 0
        vals = image[mask]
        image[mask] = (vals - vals.mean()) / max(float(vals.std()), 1e-8)
        image[~mask] = 0.0
    else:
        image = (image - image.mean()) / max(float(image.std()), 1e-8)
    return image


def _resample_data(data: np.ndarray, new_shape: Sequence[int], order: int) -> np.ndarray:
    from skimage.transform import resize

    out = np.zeros((data.shape[0], *new_shape), dtype=np.float32)
    for c in range(data.shape[0]):
        out[c] = resize(data[c].astype(np.float32), tuple(new_shape), order=order,
                        mode="edge", anti_aliasing=False)
    return out


def _resample_seg(seg: np.ndarray, new_shape: Sequence[int], order: int) -> np.ndarray:
    """Resample labels the nnU-Net way: one-hot → linear resize → argmax."""
    from skimage.transform import resize

    out = np.zeros((seg.shape[0], *new_shape), dtype=seg.dtype)
    for c in range(seg.shape[0]):
        labels = np.unique(seg[c])
        if len(labels) == 1:
            out[c] = labels[0]
            continue
        acc = np.zeros((len(labels), *new_shape), dtype=np.float32)
        for i, lab in enumerate(labels):
            acc[i] = resize((seg[c] == lab).astype(np.float32), tuple(new_shape),
                            order=order, mode="edge", anti_aliasing=False)
        out[c] = labels[acc.argmax(0)]
    return out


def compute_new_shape(
    old_shape: Sequence[int], old_spacing: Sequence[float], new_spacing: Sequence[float]
) -> List[int]:
    scale = np.array(old_spacing) / np.array(new_spacing)
    return [int(round(s)) for s in np.array(old_shape) * scale]


# --------------------------------------------------------------------------------------
# Foreground location sampling (port of DefaultPreprocessor._sample_foreground_locations)
# --------------------------------------------------------------------------------------

def sample_foreground_locations(
    seg: np.ndarray, classes: Sequence[int], seed: int = 1234, num_samples: int = 10000
) -> Dict[int, np.ndarray]:
    """Up to ``num_samples`` voxel coordinates per class (≥1 % of that class's voxels).

    Coordinates include the leading channel axis, exactly like nnU-Net, because the
    sampler indexes them as ``voxel[i + 1]``.
    """
    rng = np.random.RandomState(seed)
    class_locs: Dict[int, np.ndarray] = {}
    for c in classes:
        all_locs = np.argwhere(seg == c)
        if len(all_locs) == 0:
            class_locs[c] = np.zeros((0, seg.ndim), dtype=np.int32)
            continue
        target = min(num_samples, len(all_locs))
        target = max(target, int(np.ceil(len(all_locs) * 0.01)))
        selected = all_locs[rng.choice(len(all_locs), target, replace=False)]
        class_locs[c] = selected.astype(np.int32)
    return class_locs


# --------------------------------------------------------------------------------------
# Per-case driver
# --------------------------------------------------------------------------------------

def preprocess_case(
    image_files: Sequence[str],
    seg_file: Optional[str],
    cfg: ConfigurationPlan,
    foreground_labels: Sequence[int],
    label_map: Optional[Dict[int, int]] = None,
    transpose_forward: Sequence[int] = (0, 1, 2),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Run the full nnU-Net preprocessing chain for one case."""
    channels, per_channel_props = [], []
    for f in image_files:
        arr, props = read_image_sitk(f)
        channels.append(arr)
        per_channel_props.append(props)
    data = np.stack(channels, axis=0)
    original_spacing = per_channel_props[0]["spacing"]

    if seg_file is not None:
        seg_arr, _ = read_image_sitk(seg_file)
        seg = np.rint(seg_arr).astype(np.int8)[None]
        if label_map:
            remapped = seg.copy()
            for src, dst in label_map.items():
                if src != dst:
                    remapped[seg == src] = dst
            seg = remapped
    else:
        seg = None

    # nnU-Net transposes data into the plan's axis order before everything else.
    tf = list(transpose_forward)
    if tf != [0, 1, 2]:
        data = data.transpose([0, *(a + 1 for a in tf)])
        if seg is not None:
            seg = seg.transpose([0, *(a + 1 for a in tf)])
        original_spacing = [original_spacing[a] for a in tf]

    props: Dict = {
        "spacing": list(original_spacing),
        "shape_before_cropping": list(data.shape[1:]),
        "sitk_stuff": {k: per_channel_props[0][k] for k in
                       ("sitk_spacing", "sitk_origin", "sitk_direction")},
    }

    data, seg, bbox = crop_to_nonzero(data, seg)
    props["bbox_used_for_cropping"] = bbox
    props["shape_after_cropping_and_before_resampling"] = list(data.shape[1:])

    use_mask = cfg.use_mask_for_norm
    for c in range(data.shape[0]):
        # A 5th (subtraction) channel is not in the 4-channel plan; reuse the last entry.
        data[c] = zscore_normalize(data[c], seg, bool(use_mask[min(c, len(use_mask) - 1)]))

    target_spacing = list(cfg.spacing)
    new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
    if list(new_shape) != list(data.shape[1:]):
        data = _resample_data(data, new_shape, cfg.resampling_order_data)
        if seg is not None:
            seg = _resample_seg(seg, new_shape, cfg.resampling_order_seg)
    props["shape_after_resampling"] = list(data.shape[1:])
    props["spacing_after_resampling"] = target_spacing

    if seg is not None:
        props["class_locations"] = sample_foreground_locations(seg, foreground_labels)

    return data, (seg if seg is not None else np.zeros((1, *data.shape[1:]), np.int8)), props


def save_preprocessed_case(
    out_dir: Path, case_id: str, data: np.ndarray, seg: np.ndarray, props: Dict,
    store_dtype: str = "float16",
) -> None:
    """Write the nnU-Net *unpacked* layout: ``<case>.npy`` / ``<case>_seg.npy`` / ``<case>.pkl``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{case_id}.npy", data.astype(np.dtype(store_dtype), copy=False))
    np.save(out_dir / f"{case_id}_seg.npy", seg.astype(np.int8, copy=False))
    with open(out_dir / f"{case_id}.pkl", "wb") as f:
        pickle.dump(props, f)


def case_is_cached(out_dir: Path, case_id: str) -> bool:
    out_dir = Path(out_dir)
    return all((out_dir / f"{case_id}{suffix}").is_file()
               for suffix in (".npy", "_seg.npy", ".pkl"))
