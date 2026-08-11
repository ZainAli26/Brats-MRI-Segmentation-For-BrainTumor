"""Reader for an nnU-Net v2 ``plans.json`` — the single source of truth for the replica.

The whole point of this package is that we do NOT re-invent nnU-Net's configuration.
`nnUNetv2_plan_and_preprocess` already fingerprinted the dataset and emitted a plan
(here: ``nnUNetResEncM_11GBPlans`` for Dataset102_BraTS2024ResEnc, an 11 GB VRAM
budget). Every number the custom loop needs — patch size, batch size, spacing,
normalization scheme, the exact network topology, whether Dice is pooled over the
batch — is read straight out of that file, so the replica cannot silently drift from
the native run it is trying to reproduce.

Nothing here imports nnunetv2. The plan is plain JSON; this module just gives it a
typed surface and derives the two quantities nnU-Net computes on the fly
(``pool_op_kernel_sizes`` and ``deep_supervision_scales``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _resolve_dotted(path: str):
    """Import ``a.b.c.Name`` and return ``Name`` (used for conv_op/norm_op/nonlin)."""
    import importlib

    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


@dataclass
class ArchitecturePlan:
    """The ``architecture`` sub-dict of one configuration, with classes resolved."""

    network_class_name: str          # e.g. "...unet.ResidualEncoderUNet"
    n_stages: int
    features_per_stage: List[int]
    kernel_sizes: List[List[int]]
    strides: List[List[int]]
    n_blocks_per_stage: List[int]    # encoder blocks (nnU-Net calls this n_conv_per_stage for PlainConvUNet)
    n_conv_per_stage_decoder: List[int]
    conv_op: Any
    norm_op: Any
    norm_op_kwargs: Dict[str, Any]
    dropout_op: Any
    dropout_op_kwargs: Optional[Dict[str, Any]]
    nonlin: Any
    nonlin_kwargs: Dict[str, Any]
    conv_bias: bool

    @property
    def short_class_name(self) -> str:
        return self.network_class_name.rsplit(".", 1)[-1]

    @classmethod
    def from_legacy_configuration(cls, d: Dict[str, Any]) -> "ArchitecturePlan":
        """Adapt a pre-2.4 plans schema, which spells the architecture out inline.

        nnU-Net ≤ 2.3 stored ``UNet_class_name`` / ``UNet_base_num_features`` /
        ``n_conv_per_stage_encoder`` / ``pool_op_kernel_sizes`` at the configuration level
        instead of a nested ``architecture`` dict, and left the conv/norm/nonlin classes
        implicit in ``get_network_from_plans``. This reconstructs the same network so a
        plan emitted by either nnU-Net generation is usable — the ResEnc-M plan being
        replicated came from ≥ 2.4, while the locally installed nnU-Net is 2.1.
        """
        import torch.nn as nn

        kernel_sizes = [list(k) for k in d["conv_kernel_sizes"]]
        n_stages = len(kernel_sizes)
        dim = len(kernel_sizes[0])
        conv_op = {2: nn.Conv2d, 3: nn.Conv3d}[dim]
        norm_op = {2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[dim]

        base = int(d["UNet_base_num_features"])
        max_feat = int(d["unet_max_num_features"])
        features = [min(base * 2 ** i, max_feat) for i in range(n_stages)]
        class_name = d["UNet_class_name"]

        return cls(
            network_class_name=f"dynamic_network_architectures.architectures.unet.{class_name}",
            n_stages=n_stages,
            features_per_stage=features,
            kernel_sizes=kernel_sizes,
            strides=[list(s) for s in d["pool_op_kernel_sizes"]],
            n_blocks_per_stage=list(d["n_conv_per_stage_encoder"]),
            n_conv_per_stage_decoder=list(d["n_conv_per_stage_decoder"]),
            conv_op=conv_op,
            norm_op=norm_op,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            conv_bias=True,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchitecturePlan":
        kw = dict(d["arch_kwargs"])
        # `_kw_requires_import` lists the kwargs stored as dotted import paths.
        for key in d.get("_kw_requires_import", []):
            if kw.get(key) is not None:
                kw[key] = _resolve_dotted(kw[key])
        return cls(
            network_class_name=d["network_class_name"],
            n_stages=kw["n_stages"],
            features_per_stage=list(kw["features_per_stage"]),
            kernel_sizes=[list(k) for k in kw["kernel_sizes"]],
            strides=[list(s) for s in kw["strides"]],
            n_blocks_per_stage=list(kw["n_blocks_per_stage"]),
            n_conv_per_stage_decoder=list(kw["n_conv_per_stage_decoder"]),
            conv_op=kw["conv_op"],
            norm_op=kw["norm_op"],
            norm_op_kwargs=dict(kw.get("norm_op_kwargs") or {}),
            dropout_op=kw.get("dropout_op"),
            dropout_op_kwargs=kw.get("dropout_op_kwargs"),
            nonlin=kw["nonlin"],
            nonlin_kwargs=dict(kw.get("nonlin_kwargs") or {}),
            conv_bias=bool(kw.get("conv_bias", True)),
        )


@dataclass
class ConfigurationPlan:
    """One entry of ``plans['configurations']`` (we only ever use ``3d_fullres``)."""

    name: str
    data_identifier: str
    batch_size: int
    patch_size: List[int]
    spacing: List[float]
    normalization_schemes: List[str]
    use_mask_for_norm: List[bool]
    batch_dice: bool
    architecture: ArchitecturePlan
    resampling_order_data: int = 3
    resampling_order_seg: int = 1

    @property
    def dim(self) -> int:
        return len(self.patch_size)

    @property
    def pool_op_kernel_sizes(self) -> List[List[int]]:
        """nnU-Net's ``pool_op_kernel_sizes`` == the encoder ``strides``."""
        return self.architecture.strides

    @property
    def deep_supervision_scales(self) -> List[List[float]]:
        """Resolution of each deep-supervision head relative to the full patch.

        Mirrors ``nnUNetTrainer._get_deep_supervision_scales``: the cumulative product
        of the pooling strides, inverted, dropping the last (deepest) entry — the
        decoder emits ``n_stages - 1`` outputs, not ``n_stages``.
        """
        cumprod = np.cumprod(np.vstack(self.pool_op_kernel_sizes), axis=0)
        return [[float(v) for v in x] for x in (1 / cumprod)][:-1]

    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> "ConfigurationPlan":
        architecture = (ArchitecturePlan.from_dict(d["architecture"])
                        if "architecture" in d
                        else ArchitecturePlan.from_legacy_configuration(d))
        return cls(
            name=name,
            data_identifier=d["data_identifier"],
            batch_size=int(d["batch_size"]),
            patch_size=[int(p) for p in d["patch_size"]],
            spacing=[float(s) for s in d["spacing"]],
            normalization_schemes=list(d["normalization_schemes"]),
            use_mask_for_norm=[bool(b) for b in d["use_mask_for_norm"]],
            batch_dice=bool(d["batch_dice"]),
            architecture=architecture,
            resampling_order_data=int(d.get("resampling_fn_data_kwargs", {}).get("order", 3)),
            resampling_order_seg=int(d.get("resampling_fn_seg_kwargs", {}).get("order", 1)),
        )


@dataclass
class Plans:
    """A parsed nnU-Net ``plans.json``."""

    dataset_name: str
    plans_name: str
    transpose_forward: List[int]
    transpose_backward: List[int]
    configurations: Dict[str, ConfigurationPlan]
    foreground_intensity_properties_per_channel: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Plans":
        with open(Path(path).expanduser()) as f:
            raw = json.load(f)  # tolerates the NaN/Infinity nnU-Net writes for MRI stats
        return cls(
            dataset_name=raw["dataset_name"],
            plans_name=raw["plans_name"],
            transpose_forward=list(raw.get("transpose_forward", [0, 1, 2])),
            transpose_backward=list(raw.get("transpose_backward", [0, 1, 2])),
            configurations={
                k: ConfigurationPlan.from_dict(k, v) for k, v in raw["configurations"].items()
            },
            foreground_intensity_properties_per_channel=raw.get(
                "foreground_intensity_properties_per_channel", {}
            ),
            raw=raw,
        )

    def get_configuration(self, name: str = "3d_fullres") -> ConfigurationPlan:
        if name not in self.configurations:
            raise KeyError(
                f"Configuration '{name}' not in plans '{self.plans_name}'. "
                f"Available: {sorted(self.configurations)}"
            )
        return self.configurations[name]

    def summary(self, configuration: str = "3d_fullres") -> str:
        c = self.get_configuration(configuration)
        a = c.architecture
        return (
            f"{self.plans_name} / {configuration} ({self.dataset_name})\n"
            f"  patch {c.patch_size}  batch {c.batch_size}  spacing {c.spacing}  "
            f"batch_dice {c.batch_dice}\n"
            f"  {a.short_class_name}: {a.n_stages} stages, features {a.features_per_stage},\n"
            f"    encoder blocks {a.n_blocks_per_stage}, decoder convs {a.n_conv_per_stage_decoder},\n"
            f"    strides {a.strides}\n"
            f"  deep supervision scales {[[round(v, 4) for v in s] for s in c.deep_supervision_scales]}"
        )
