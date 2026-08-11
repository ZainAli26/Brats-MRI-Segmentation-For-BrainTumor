"""Build the planned network — architecture *and* initialisation — straight from plans.json.

Two things distinguish this from ``src/models/factory.py``:

1. **Nothing is hand-typed.** Stage count, features, kernels, strides, per-stage block
   counts, norm/nonlinearity all come out of ``plans['configurations'][cfg]['architecture']``,
   so the replica's network is the network the planner chose for the 11 GB budget.

2. **nnU-Net's initialisation is applied.** ``get_network_from_plans`` ends with
   ``model.apply(InitWeights_He(1e-2))`` and, for residual encoders,
   ``init_last_bn_before_add_to_0``. PyTorch's default Conv3d init is
   ``kaiming_uniform(a=sqrt(5))`` — a much smaller gain — and the residual branches start
   *non*-zero. Training a 100 M-parameter ResEnc from that default with SGD at lr 1e-2 is
   exactly the regime where the old loop collapsed to all-background, so this is not a
   cosmetic detail.

Gradient checkpointing on the encoder is optional and off by default; it is what lets the
[128, 192, 128] patch fit an 8 GB card at the cost of ~25 % step time.
"""

from __future__ import annotations

import types
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from src.nnunet_replica.plans import ConfigurationPlan


class InitWeights_He:
    """Port of ``nnunetv2.utilities.network_initialization.InitWeights_He``."""

    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def init_last_norm_before_add_to_0(module: nn.Module) -> None:
    """Zero the last norm in each residual block so blocks start as identity."""
    from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD

    if isinstance(module, BasicBlockD):
        module.conv2.norm.weight = nn.init.constant_(module.conv2.norm.weight, 0)
        module.conv2.norm.bias = nn.init.constant_(module.conv2.norm.bias, 0)
    if isinstance(module, BottleneckD):
        module.conv3.norm.weight = nn.init.constant_(module.conv3.norm.weight, 0)
        module.conv3.norm.bias = nn.init.constant_(module.conv3.norm.bias, 0)


def enable_encoder_grad_checkpointing(model: nn.Module) -> nn.Module:
    """Recompute encoder-stage activations in the backward pass instead of storing them."""
    from torch.utils.checkpoint import checkpoint

    enc = getattr(model, "encoder", None)
    if enc is None or not hasattr(enc, "stages"):
        raise RuntimeError("grad_checkpointing requested but model.encoder.stages not found")

    def ckpt_forward(self, x):
        if getattr(self, "stem", None) is not None:
            x = self.stem(x)
        out = []
        for stage in self.stages:
            x = checkpoint(stage, x, use_reentrant=False)
            out.append(x)
        return out if self.return_skips else out[-1]

    enc.forward = types.MethodType(ckpt_forward, enc)
    return model


def build_network(
    cfg: ConfigurationPlan,
    num_input_channels: int,
    num_output_classes: int,
    deep_supervision: bool = True,
    grad_checkpointing: bool = False,
) -> nn.Module:
    """Instantiate the planned architecture with nnU-Net's initialisation."""
    from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet

    arch = cfg.architecture
    mapping = {"PlainConvUNet": PlainConvUNet, "ResidualEncoderUNet": ResidualEncoderUNet}
    if arch.short_class_name not in mapping:
        raise ValueError(
            f"Unsupported network_class_name '{arch.network_class_name}'. "
            f"Supported: {sorted(mapping)}"
        )
    network_class = mapping[arch.short_class_name]

    # PlainConvUNet calls the encoder depth 'n_conv_per_stage'; ResEnc calls it 'n_blocks_per_stage'.
    blocks_kw = (
        {"n_blocks_per_stage": arch.n_blocks_per_stage}
        if network_class is ResidualEncoderUNet
        else {"n_conv_per_stage": arch.n_blocks_per_stage}
    )

    model = network_class(
        input_channels=num_input_channels,
        n_stages=arch.n_stages,
        features_per_stage=arch.features_per_stage,
        conv_op=arch.conv_op,
        kernel_sizes=arch.kernel_sizes,
        strides=arch.strides,
        num_classes=num_output_classes,
        deep_supervision=deep_supervision,
        conv_bias=arch.conv_bias,
        norm_op=arch.norm_op,
        norm_op_kwargs=arch.norm_op_kwargs,
        dropout_op=arch.dropout_op,
        dropout_op_kwargs=arch.dropout_op_kwargs,
        nonlin=arch.nonlin,
        nonlin_kwargs=arch.nonlin_kwargs,
        n_conv_per_stage_decoder=arch.n_conv_per_stage_decoder,
        **blocks_kw,
    )

    model.apply(InitWeights_He(1e-2))
    if network_class is ResidualEncoderUNet:
        model.apply(init_last_norm_before_add_to_0)

    if grad_checkpointing:
        model = enable_encoder_grad_checkpointing(model)
    return model


def build_segresnet_ds(
    num_input_channels: int,
    num_output_classes: int,
    init_filters: int = 32,
    blocks_down: Sequence[int] = (1, 2, 2, 4, 4),
    dsdepth: int = 5,
) -> Tuple[nn.Module, List[List[float]]]:
    """Architecture-ablation escape hatch: MONAI ``SegResNetDS`` in the same recipe.

    Everything else about the loop is unchanged — the patch, batch, sampling, augmentation,
    optimiser, schedule and loss all still come from the plan — so a run using this
    isolates "which network" from "which training procedure".

    Returns the network *and* its own deep-supervision scales, since they come from the
    net's downsampling stages rather than the plan's. ``SegResNetDS`` emits outputs
    highest-resolution first, halving each time — but only from its *upsampling* stages,
    so it can never emit more than ``len(blocks_down) - 1`` heads no matter what
    ``dsdepth`` asks for. Clamping here (rather than letting MONAI do it silently) keeps
    the returned scale list equal to the real head count, which is what the loss weights
    and the target-downsampling transform are both built from.
    """
    from monai.networks.nets import SegResNetDS

    max_heads = max(1, len(blocks_down) - 1)
    requested = max(1, int(dsdepth))
    dsdepth = min(requested, max_heads)
    model = SegResNetDS(
        spatial_dims=3,
        init_filters=init_filters,
        in_channels=num_input_channels,
        out_channels=num_output_classes,
        blocks_down=list(blocks_down),
        dsdepth=dsdepth,
    )
    scales = [[1 / (2 ** i)] * 3 for i in range(dsdepth)]
    return model, scales


def set_deep_supervision_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle DS — nnU-Net turns it off for final full-image validation.

    ``SegResNetDS`` has no such switch: it returns every head in train mode and only the
    full-resolution one in eval mode, which is already the behaviour we want, so this is
    a no-op there.
    """
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "deep_supervision"):
        decoder.deep_supervision = enabled


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
