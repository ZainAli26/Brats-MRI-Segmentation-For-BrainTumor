"""Shared preprocessing pipeline for all models.

Keeps preprocessing identical across architectures so results are comparable.
Uses MONAI transforms for reproducibility.
"""

from typing import Dict, List

from monai import transforms as T


def get_train_transforms(
    spatial_size: List[int],
    modalities: List[str],
    label_map: Dict[int, int],
    aug_config: dict,
) -> T.Compose:
    """Training transforms: preprocessing + augmentation."""
    keys_img = [f"image_{m}" for m in modalities]
    keys_all = keys_img + ["label"]

    transform_list = [
        # Load NIfTI files
        T.LoadImaged(keys=keys_all, image_only=True),
        T.EnsureChannelFirstd(keys=keys_all),
        # Remap labels: {0:0, 1:1, 2:2, 4:3} for contiguous classes
        _RemapLabelsd(keys=["label"], label_map=label_map),
        # Stack modalities into single 4-channel image
        _StackModalitiesd(modality_keys=keys_img, output_key="image"),
        # Orientation & spacing
        T.Orientationd(keys=["image", "label"], axcodes="RAS"),
        T.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest")),
        # Intensity normalization: z-score on nonzero voxels per channel
        T.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        # Crop foreground + resize
        T.CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        T.SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        T.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=3, neg=1,
            num_samples=1,
        ),
        # Data augmentation
        T.RandFlipd(keys=["image", "label"], prob=aug_config["random_flip_prob"], spatial_axis=0),
        T.RandFlipd(keys=["image", "label"], prob=aug_config["random_flip_prob"], spatial_axis=1),
        T.RandFlipd(keys=["image", "label"], prob=aug_config["random_flip_prob"], spatial_axis=2),
        T.RandRotate90d(keys=["image", "label"], prob=aug_config.get("random_rotate_prob", 0.3), max_k=3),
        T.RandShiftIntensityd(keys=["image"], offsets=aug_config["random_intensity_shift"], prob=0.5),
        T.RandScaleIntensityd(keys=["image"], factors=aug_config["random_intensity_scale"], prob=0.5),
        T.ToTensord(keys=["image", "label"]),
    ]
    return T.Compose(transform_list)


def get_val_transforms(
    spatial_size: List[int],
    modalities: List[str],
    label_map: Dict[int, int],
) -> T.Compose:
    """Validation/test transforms: preprocessing only, no augmentation."""
    keys_img = [f"image_{m}" for m in modalities]
    keys_all = keys_img + ["label"]

    transform_list = [
        T.LoadImaged(keys=keys_all, image_only=True),
        T.EnsureChannelFirstd(keys=keys_all),
        _RemapLabelsd(keys=["label"], label_map=label_map),
        _StackModalitiesd(modality_keys=keys_img, output_key="image"),
        T.Orientationd(keys=["image", "label"], axcodes="RAS"),
        T.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest")),
        T.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        T.CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        T.SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        T.ToTensord(keys=["image", "label"]),
    ]
    return T.Compose(transform_list)


def get_inference_transforms(
    spatial_size: List[int],
    modalities: List[str],
) -> T.Compose:
    """Inference transforms for data without labels."""
    keys_img = [f"image_{m}" for m in modalities]

    transform_list = [
        T.LoadImaged(keys=keys_img, image_only=True),
        T.EnsureChannelFirstd(keys=keys_img),
        _StackModalitiesd(modality_keys=keys_img, output_key="image"),
        T.Orientationd(keys=["image"], axcodes="RAS"),
        T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        T.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        T.CropForegroundd(keys=["image"], source_key="image", margin=10),
        T.SpatialPadd(keys=["image"], spatial_size=spatial_size),
        T.ToTensord(keys=["image"]),
    ]
    return T.Compose(transform_list)


class _RemapLabelsd(T.MapTransform):
    """Remap label values (e.g., {4 -> 3} for contiguous classes)."""

    def __init__(self, keys, label_map: Dict[int, int]):
        super().__init__(keys)
        self.label_map = label_map

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = d[key].clone() if hasattr(d[key], 'clone') else d[key].copy()
            for src, dst in self.label_map.items():
                if src != dst:
                    result[d[key] == src] = dst
            d[key] = result
        return d


class _StackModalitiesd(T.MapTransform):
    """Stack individual modality images into a single multi-channel tensor."""

    def __init__(self, modality_keys: List[str], output_key: str = "image"):
        super().__init__(modality_keys)
        self.modality_keys = modality_keys
        self.output_key = output_key

    def __call__(self, data):
        import torch
        d = dict(data)
        channels = []
        for key in self.modality_keys:
            img = d[key]
            if hasattr(img, 'shape') and len(img.shape) == 4:
                channels.append(img[0])  # Remove channel dim from each
            else:
                channels.append(img)
        if hasattr(channels[0], 'numpy'):
            import numpy as np
            d[self.output_key] = np.stack(channels, axis=0)
        else:
            d[self.output_key] = torch.stack(channels, dim=0)
        # Remove individual modality keys
        for key in self.modality_keys:
            del d[key]
        return d
