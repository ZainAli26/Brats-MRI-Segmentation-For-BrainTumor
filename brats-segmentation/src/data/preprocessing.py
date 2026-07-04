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
    """Training transforms: preprocessing + augmentation.

    aug_config["preset"] == "nnunet" selects a full nnU-Net v2-style augmentation
    set (affine rotation/scaling, Gaussian noise/blur, brightness/contrast, gamma,
    mirroring); any other value keeps the original light set, so existing experiments
    are unchanged. Foreground oversampling is controlled by aug_config["crop_pos"] /
    ["crop_neg"] (default 3/1; nnU-Net uses ~1/2 -> 33% foreground).
    """
    keys_img = [f"image_{m}" for m in modalities]
    keys_all = keys_img + ["label"]

    preset = aug_config.get("preset", "default")
    # `or` guards the overfit path, which zeroes every aug value (incl. these).
    pos = aug_config.get("crop_pos", 3) or 3
    neg = aug_config.get("crop_neg", 1) or 1

    # Deterministic preprocessing + foreground-biased patch sampling (shared).
    common = [
        T.LoadImaged(keys=keys_all, image_only=True),
        T.EnsureChannelFirstd(keys=keys_all),
        # Remap labels per config label_map (identity for already-contiguous schemes:
        # BraTS 2024 = 0..4 NETC/SNFH/ET/RC; BraTS 2023 = 0..3 NCR/ED/ET)
        _RemapLabelsd(keys=["label"], label_map=label_map),
        _StackModalitiesd(modality_keys=keys_img, output_key="image"),
        T.Orientationd(keys=["image", "label"], axcodes="RAS"),
        T.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                   mode=("bilinear", "nearest")),
        T.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        T.CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        T.SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        T.RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=spatial_size, pos=pos, neg=neg, num_samples=1,
        ),
    ]

    if preset == "nnunet":
        # nnU-Net v2 default augmentation. Spatial transforms list image+label so they
        # share the same random draw (geometry stays aligned across all modalities).
        aug = [
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            T.RandAffined(
                keys=["image", "label"], prob=0.2,
                rotate_range=(0.5236, 0.5236, 0.5236),  # +/- 30 deg
                scale_range=(0.3, 0.3, 0.3),            # ~0.7 - 1.3
                mode=("bilinear", "nearest"), padding_mode="zeros",
            ),
            # Brightness/contrast: MONAI built-in per-channel support.
            T.RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.15, channel_wise=True),
            T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15, channel_wise=True),
            # Per-channel noise + blur + SimulateLowResolution + gamma, vectorized
            # (each modality augmented independently — nnU-Net p_per_channel).
            _PerChannelIntensityd(
                key="image",
                noise_prob=0.15, noise_std=0.1,
                blur_prob=0.2, sigma_range=(0.5, 1.5),
                lowres_prob=0.25, zoom_range=(0.5, 1.0),
                gamma_prob=0.15, gamma_range=(0.7, 1.5),
            ),
        ]
    else:
        aug = [
            T.RandFlipd(keys=["image", "label"], prob=aug_config["random_flip_prob"], spatial_axis=0),
            T.RandFlipd(keys=["image", "label"], prob=aug_config["random_flip_prob"], spatial_axis=1),
            T.RandFlipd(keys=["image", "label"], prob=aug_config["random_flip_prob"], spatial_axis=2),
            T.RandRotate90d(keys=["image", "label"], prob=aug_config.get("random_rotate_prob", 0.3), max_k=3),
            T.RandShiftIntensityd(keys=["image"], offsets=aug_config["random_intensity_shift"], prob=0.5),
            T.RandScaleIntensityd(keys=["image"], factors=aug_config["random_intensity_scale"], prob=0.5),
        ]

    return T.Compose(common + aug + [T.ToTensord(keys=["image", "label"])])


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


class _PerChannelIntensityd(T.MapTransform, T.RandomizableTransform):
    """Vectorized per-channel intensity augmentation (nnU-Net p_per_channel).

    Replaces a per-channel Python loop over noise/blur/low-res/gamma with batched
    torch ops: each modality channel rolls its OWN probability and parameters, but
    all channels are processed together (grouped separable conv for blur; pointwise
    math for noise/gamma). Low-res keeps a minimal loop over only the selected
    channels because per-channel zoom changes the spatial size.

    Operates on the stacked image tensor `key` of shape (C, H, W, D). Image only —
    never touched the label. Probabilities/ranges mirror nnU-Net's defaults.
    """

    def __init__(
        self, key: str = "image",
        noise_prob: float = 0.15, noise_std: float = 0.1,
        blur_prob: float = 0.2, sigma_range=(0.5, 1.5),
        lowres_prob: float = 0.25, zoom_range=(0.5, 1.0),
        gamma_prob: float = 0.15, gamma_range=(0.7, 1.5),
    ):
        T.MapTransform.__init__(self, [key])
        T.RandomizableTransform.__init__(self, prob=1.0)
        self.key = key
        self.noise_prob, self.noise_std = noise_prob, noise_std
        self.blur_prob, self.sigma_range = blur_prob, sigma_range
        self.lowres_prob, self.zoom_range = lowres_prob, zoom_range
        self.gamma_prob, self.gamma_range = gamma_prob, gamma_range

    @staticmethod
    def _gaussian_kernel(sigma: float, radius: int):
        import torch
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return k / k.sum()

    def _blur(self, img):
        """Separable grouped-conv Gaussian blur with an independent sigma per channel."""
        import torch
        import torch.nn.functional as F
        C = img.shape[0]
        sel = self.R.uniform(size=C) < self.blur_prob
        if not sel.any():
            return img
        radius = max(1, int(round(3 * self.sigma_range[1])))
        K = 2 * radius + 1
        kernels = []  # per-channel 1D kernel (delta if not selected -> identity)
        for c in range(C):
            if sel[c]:
                sigma = float(self.R.uniform(*self.sigma_range))
                kernels.append(self._gaussian_kernel(sigma, radius))
            else:
                delta = torch.zeros(K); delta[radius] = 1.0
                kernels.append(delta)
        base = torch.stack(kernels, dim=0).to(img.dtype)  # (C, K)
        x = img.unsqueeze(0)  # (1, C, H, W, D)
        for axis in range(3):
            shape = [C, 1, 1, 1, 1]; shape[axis + 2] = K
            w = base.view(C, K)[:, None, :].reshape(shape)
            pad = [0, 0, 0]; pad[axis] = radius
            x = F.conv3d(x, w, groups=C, padding=(pad[0], pad[1], pad[2]))
        return x[0]

    def _lowres(self, img):
        """Down- then up-sample per selected channel (different zoom each)."""
        import torch
        import torch.nn.functional as F
        C = img.shape[0]
        full = img.shape[1:]
        for c in range(C):
            if self.R.uniform() >= self.lowres_prob:
                continue
            zoom = float(self.R.uniform(*self.zoom_range))
            low = [max(1, int(round(s * zoom))) for s in full]
            ch = img[c][None, None]  # (1,1,H,W,D)
            down = F.interpolate(ch, size=low, mode="nearest")
            up = F.interpolate(down, size=tuple(full), mode="trilinear", align_corners=False)
            img[c] = up[0, 0]
        return img

    def _gamma(self, img):
        """Per-channel gamma on min-max-normalized intensities."""
        C = img.shape[0]
        for c in range(C):
            if self.R.uniform() >= self.gamma_prob:
                continue
            g = float(self.R.uniform(*self.gamma_range))
            ch = img[c]
            mn = ch.min(); rng = ch.max() - mn
            if rng > 1e-6:
                img[c] = ((ch - mn) / rng) ** g * rng + mn
        return img

    def __call__(self, data):
        import torch
        d = dict(data)
        img = torch.as_tensor(d[self.key]).clone().float()  # (C, H, W, D)
        C = img.shape[0]

        # Gaussian noise — fully vectorized: per-channel selection + per-channel std.
        sel = torch.as_tensor(self.R.uniform(size=C) < self.noise_prob)
        if sel.any():
            stds = torch.as_tensor(self.R.uniform(0, self.noise_std, size=C), dtype=img.dtype)
            noise = torch.randn_like(img) * stds[:, None, None, None]
            img = img + noise * sel[:, None, None, None].to(img.dtype)

        img = self._blur(img)
        img = self._lowres(img)
        img = self._gamma(img)

        # write back, preserving the original container type/metadata where possible
        orig = d[self.key]
        if hasattr(orig, "copy_meta_from"):
            new = orig.__class__(img)
            new.copy_meta_from(orig)
            d[self.key] = new
        else:
            d[self.key] = img
        return d


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
