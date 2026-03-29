"""Model factory for BraTS segmentation architectures.

Supports:
  - nnunet_v2:  Real nnU-Net v2 PlainConvUNet / ResidualEncoderUNet
  - dynunet:    MONAI's DynUNet (nnU-Net-style architecture)
  - swin_unetr: MONAI SwinUNETR
  - segresnet:  MONAI SegResNet

All models plug into the same shared preprocessing and training loop.
"""

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet, SwinUNETR, SegResNet
from rich.console import Console

console = Console()


def _create_nnunet_v2(in_ch: int, out_ch: int, cfg: dict) -> nn.Module:
    """Create the actual nnU-Net v2 PlainConvUNet architecture.

    This uses the real nnU-Net v2 network class, not a MONAI approximation.
    The architecture is configured to match nnU-Net's auto-planned layout
    for 128^3 BraTS inputs.
    """
    from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet

    variant = cfg.get("variant", "plain")  # "plain" or "residual"

    # nnU-Net v2 expects these as nested lists
    kernel_sizes = cfg["kernel_sizes"]
    strides = cfg["strides"]
    n_stages = len(kernel_sizes)
    features_per_stage = cfg["features_per_stage"]

    # nnU-Net v2 conv kwargs
    conv_kwargs = {"kernel_size": 3, "stride": 1, "padding": 1, "bias": True}
    norm_kwargs = {"eps": 1e-5, "affine": True}

    if variant == "residual":
        model = ResidualEncoderUNet(
            input_channels=in_ch,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv3d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            num_classes=out_ch,
            n_blocks_per_stage=[cfg.get("n_blocks_encoder", 2)] * n_stages,
            n_conv_per_stage_decoder=[cfg.get("n_blocks_decoder", 2)] * (n_stages - 1),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs=norm_kwargs,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=cfg.get("deep_supervision", True),
        )
        console.print(f"[green]Created nnU-Net v2 ResidualEncoderUNet with features={features_per_stage}[/green]")
    else:
        model = PlainConvUNet(
            input_channels=in_ch,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv3d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            num_classes=out_ch,
            n_conv_per_stage=[cfg.get("n_blocks_encoder", 2)] * n_stages,
            n_conv_per_stage_decoder=[cfg.get("n_blocks_decoder", 2)] * (n_stages - 1),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs=norm_kwargs,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=cfg.get("deep_supervision", True),
        )
        console.print(f"[green]Created nnU-Net v2 PlainConvUNet with features={features_per_stage}[/green]")

    return model


def create_model(config: dict) -> nn.Module:
    """Create a segmentation model based on config.

    Args:
        config: Full config dict with 'model' section.

    Returns:
        PyTorch model instance.
    """
    model_cfg = config["model"]
    name = model_cfg["name"].lower()
    in_ch = model_cfg["in_channels"]
    out_ch = model_cfg["out_channels"]

    if name == "nnunet_v2":
        cfg = model_cfg["nnunet_v2"]
        model = _create_nnunet_v2(in_ch, out_ch, cfg)

    elif name == "dynunet":
        cfg = model_cfg["dynunet"]
        model = DynUNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=cfg["kernel_sizes"],
            strides=cfg["strides"],
            upsample_kernel_size=cfg["strides"][1:],
            filters=cfg["filters"],
            deep_supervision=cfg["deep_supervision"],
            deep_supr_num=cfg.get("deep_supervision_heads", 3),
        )
        console.print(f"[green]Created DynUNet (MONAI nnU-Net) with filters={cfg['filters']}[/green]")

    elif name == "swin_unetr":
        cfg = model_cfg["swin_unetr"]
        model = SwinUNETR(
            in_channels=in_ch,
            out_channels=out_ch,
            feature_size=cfg["feature_size"],
            depths=tuple(cfg["depths"]),
            num_heads=tuple(cfg["num_heads"]),
            drop_rate=cfg["drop_rate"],
            attn_drop_rate=cfg["attn_drop_rate"],
        )
        console.print(f"[green]Created SwinUNETR with feature_size={cfg['feature_size']}[/green]")

    elif name == "segresnet":
        cfg = model_cfg["segresnet"]
        model = SegResNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            init_filters=cfg["init_filters"],
            blocks_down=cfg["blocks_down"],
            blocks_up=cfg["blocks_up"],
            dropout_prob=cfg["dropout_prob"],
        )
        console.print(f"[green]Created SegResNet with init_filters={cfg['init_filters']}[/green]")

    else:
        raise ValueError(f"Unknown model: {name}. Choose from: nnunet_v2, dynunet, swin_unetr, segresnet")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[dim]Trainable parameters: {n_params:,}[/dim]")

    return model
