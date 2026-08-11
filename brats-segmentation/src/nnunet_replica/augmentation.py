"""nnU-Net v2's training-time augmentation, transform for transform.

This is a direct transcription of ``nnUNetTrainer.get_training_transforms`` /
``get_validation_transforms`` and ``configure_rotation_dummyDA_mirroring_and_inital_patch_size``.
It is built on **batchgenerators** — a general-purpose augmentation library, not part of
the nnU-Net engine — so the pixel-level behaviour (rotation on a coordinate grid,
order-3 data / order-1 seg interpolation, per-channel probability semantics) is
identical rather than approximated by a MONAI look-alike. The two small nnU-Net-owned
transforms (``MaskTransform`` and ``DownsampleSegForDSTransform2``) are ported below so
nothing here imports ``nnunetv2``.

The probabilities are load-bearing and differ from the hand-rolled "nnunet preset" in
``src/data/preprocessing.py`` — e.g. Gaussian noise is 0.1 (not 0.15), gamma is applied
twice (inverted p=0.1 then normal p=0.3, both ``retain_stats``), blur and low-res roll a
second coin *per channel* at 0.5. Getting these wrong is one of the ways a replica
quietly under-performs the original.

Note the sampling flow: the dataloader crops an **inflated** patch
(``compute_initial_patch_size``: [243, 270, 205] for a [128, 192, 128] plan), the spatial
transform rotates/scales inside it, and the result is centre-cropped to the real patch
size. That is what keeps rotated patches free of zero-padding at the border.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor,
    RemoveLabelTransform,
    RenameTransform,
)

ANISO_THRESHOLD = 3  # nnunetv2.configuration.ANISO_THRESHOLD


# --------------------------------------------------------------------------------------
# Ported nnU-Net-specific transforms
# --------------------------------------------------------------------------------------

class MaskTransform(AbstractTransform):
    """Zero the image wherever the seg is < 0 (outside the brain after cropping).

    Re-imposes the "outside stays exactly 0" invariant that ``use_mask_for_norm``
    established in preprocessing but that intensity augmentation has since broken.
    """

    def __init__(self, apply_to_channels: List[int], mask_idx_in_seg: int = 0,
                 set_outside_to: float = 0, data_key: str = "data", seg_key: str = "seg"):
        self.apply_to_channels = apply_to_channels
        self.mask_idx_in_seg = mask_idx_in_seg
        self.set_outside_to = set_outside_to
        self.data_key = data_key
        self.seg_key = seg_key

    def __call__(self, **data_dict):
        mask = data_dict[self.seg_key][:, self.mask_idx_in_seg] < 0
        for c in self.apply_to_channels:
            data_dict[self.data_key][:, c][mask] = self.set_outside_to
        return data_dict


class DownsampleSegForDSTransform(AbstractTransform):
    """Turn the target into one label map per deep-supervision head.

    nnU-Net supervises each decoder head at its OWN resolution by shrinking the label
    map (nearest neighbour, order 0), never by upsampling the prediction.
    """

    def __init__(self, ds_scales: Sequence, order: int = 0,
                 input_key: str = "target", output_key: str = "target"):
        self.ds_scales = ds_scales
        self.order = order
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, **data_dict):
        from batchgenerators.augmentations.utils import resize_segmentation

        seg = data_dict[self.input_key]
        axes = list(range(2, len(seg.shape)))
        output = []
        for s in self.ds_scales:
            if not isinstance(s, (tuple, list)):
                s = [s] * len(axes)
            if all(i == 1 for i in s):
                output.append(seg)
                continue
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            out_seg = np.zeros(new_shape, dtype=seg.dtype)
            for b in range(seg.shape[0]):
                for c in range(seg.shape[1]):
                    out_seg[b, c] = resize_segmentation(seg[b, c], new_shape[2:], self.order)
            output.append(out_seg)
        data_dict[self.output_key] = output
        return data_dict


# --------------------------------------------------------------------------------------
# Rotation / mirroring / initial patch size
# --------------------------------------------------------------------------------------

def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    """Inflated sampling patch, so rotation never pulls in padding. Port of nnU-Net's."""
    from batchgenerators.augmentations.utils import rotate_coords_2d, rotate_coords_3d

    rot_x = max(np.abs(rot_x)) if isinstance(rot_x, (tuple, list)) else rot_x
    rot_y = max(np.abs(rot_y)) if isinstance(rot_y, (tuple, list)) else rot_y
    rot_z = max(np.abs(rot_z)) if isinstance(rot_z, (tuple, list)) else rot_z
    half_pi = 90 / 360 * 2.0 * np.pi
    rot_x, rot_y, rot_z = min(half_pi, rot_x), min(half_pi, rot_y), min(half_pi, rot_z)

    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    else:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)


def configure_rotation_and_initial_patch_size(patch_size: Sequence[int]):
    """Port of ``configure_rotation_dummyDA_mirroring_and_inital_patch_size`` (3D only).

    Returns ``(rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes)``.
    """
    patch_size = list(patch_size)
    assert len(patch_size) == 3, "the replica trainer is 3D-only"

    do_dummy_2d = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
    if do_dummy_2d:
        rotation = {"x": (-np.pi, np.pi), "y": (0, 0), "z": (0, 0)}
    else:
        r = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
        rotation = {"x": r, "y": r, "z": r}
    mirror_axes = (0, 1, 2)

    initial_patch_size = get_patch_size(patch_size, *rotation.values(), (0.85, 1.25))
    if do_dummy_2d:
        initial_patch_size[0] = patch_size[0]
    return rotation, do_dummy_2d, [int(v) for v in initial_patch_size], mirror_axes


# --------------------------------------------------------------------------------------
# Transform pipelines
# --------------------------------------------------------------------------------------

def get_training_transforms(
    patch_size: Sequence[int],
    rotation_for_DA: dict,
    deep_supervision_scales: Optional[Sequence],
    mirror_axes: Optional[Tuple[int, ...]],
    do_dummy_2d_data_aug: bool,
    use_mask_for_norm: Optional[Sequence[bool]] = None,
    order_resampling_data: int = 3,
    order_resampling_seg: int = 1,
    border_val_seg: int = -1,
) -> AbstractTransform:
    """Exactly ``nnUNetTrainer.get_training_transforms`` (non-cascade, non-region)."""
    from batchgenerators.transforms.abstract_transforms import Compose as _Compose

    tr: List[AbstractTransform] = []

    if do_dummy_2d_data_aug:
        from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import (  # noqa: E501
            Convert2DTo3DTransform, Convert3DTo2DTransform,
        )
        ignore_axes = (0,)
        tr.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    tr.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True,
        angle_x=rotation_for_DA["x"], angle_y=rotation_for_DA["y"], angle_z=rotation_for_DA["z"],
        p_rot_per_axis=1,
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
        random_crop=False,  # the dataloader already picked the location
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False,
    ))

    if do_dummy_2d_data_aug:
        from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import (  # noqa: E501
            Convert2DTo3DTransform,
        )
        tr.append(Convert2DTo3DTransform())

    tr.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr.append(GaussianBlurTransform((0.5, 1.0), different_sigma_per_channel=True,
                                    p_per_sample=0.2, p_per_channel=0.5))
    tr.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr.append(SimulateLowResolutionTransform(
        zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
        order_downsample=0, order_upsample=3, p_per_sample=0.25, ignore_axes=ignore_axes,
    ))
    tr.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr.append(MirrorTransform(mirror_axes))

    if use_mask_for_norm is not None and any(use_mask_for_norm):
        tr.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                mask_idx_in_seg=0, set_outside_to=0))

    tr.append(RemoveLabelTransform(-1, 0))
    tr.append(RenameTransform("seg", "target", True))

    if deep_supervision_scales is not None:
        tr.append(DownsampleSegForDSTransform(deep_supervision_scales, 0,
                                              input_key="target", output_key="target"))
    tr.append(NumpyToTensor(["data", "target"], "float"))
    return _Compose(tr)


def get_validation_transforms(deep_supervision_scales: Optional[Sequence]) -> AbstractTransform:
    """Exactly ``nnUNetTrainer.get_validation_transforms``: no augmentation, DS targets only."""
    val: List[AbstractTransform] = [
        RemoveLabelTransform(-1, 0),
        RenameTransform("seg", "target", True),
    ]
    if deep_supervision_scales is not None:
        val.append(DownsampleSegForDSTransform(deep_supervision_scales, 0,
                                               input_key="target", output_key="target"))
    val.append(NumpyToTensor(["data", "target"], "float"))
    return Compose(val)


def mask_channels_for_norm(use_mask_for_norm: Sequence[bool], num_channels: int) -> List[bool]:
    """Extend/trim the plan's per-channel mask flags to the actual channel count.

    Lets a 5-channel experiment (the T1Gd−T1 subtraction channel) reuse a 4-channel plan;
    the extra channel inherits the last flag.
    """
    flags = list(use_mask_for_norm)
    if len(flags) >= num_channels:
        return flags[:num_channels]
    return flags + [flags[-1]] * (num_channels - len(flags))
