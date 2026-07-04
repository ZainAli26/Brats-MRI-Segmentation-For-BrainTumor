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


def _expand_blocks(value, n_stages: int, name: str) -> list:
    """Normalize a block-count spec into a per-stage list.

    Accepts either a scalar (broadcast to every stage, the exp01-19 behavior)
    or an explicit per-stage list/tuple (e.g. nnU-Net ResEnc's asymmetric deep
    encoder ``[1, 3, 4, 6, 6, 6]``). A list must have exactly ``n_stages`` entries.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != n_stages:
            raise ValueError(
                f"{name} list has {len(value)} entries but n_stages={n_stages}; "
                f"provide one block count per stage."
            )
        return [int(v) for v in value]
    return [int(value)] * n_stages


def _enable_grad_checkpointing(model: nn.Module) -> nn.Module:
    """Wrap the encoder stages in gradient checkpointing to cut activation memory.

    The encoder holds most of the activation memory (high-res early stages). With
    checkpointing each stage's internal activations are recomputed in the backward
    pass instead of being stored, trading ~20-30% compute for a large memory drop —
    this is what lets Adam + the 160x192x128 patch fit an 8 GB GPU.
    """
    import types
    from torch.utils.checkpoint import checkpoint

    enc = getattr(model, "encoder", None)
    if enc is None or not hasattr(enc, "stages"):
        console.print("[yellow]grad_checkpointing requested but encoder.stages not found — skipping[/yellow]")
        return model

    def ckpt_forward(self, x):
        if getattr(self, "stem", None) is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = checkpoint(s, x, use_reentrant=False)
            ret.append(x)
        return ret if self.return_skips else ret[-1]

    enc.forward = types.MethodType(ckpt_forward, enc)
    console.print("[green]Gradient checkpointing enabled on encoder stages[/green]")
    return model


def _enable_segresnet_grad_checkpointing(model: nn.Module) -> nn.Module:
    """Gradient checkpointing for MONAI SegResNet (which exposes no flag of its own).

    MONAI's SegResNet keeps full-resolution, high-channel feature maps in the down/up
    res-block stages, so at the 160x192x128 patch its activations OOM an 8 GB GPU even
    though it has far fewer params than the nnU-Net models. We override encode/decode
    (replicating MONAI's exact logic) to recompute each stage's internal activations in
    the backward pass instead of storing them — ~25% slower, large memory drop. The
    stage OUTPUTS (skip connections) are still kept; only the within-stage block
    activations are recomputed.
    """
    import types
    from torch.utils.checkpoint import checkpoint

    if not (hasattr(model, "down_layers") and hasattr(model, "up_layers")):
        console.print("[yellow]grad_checkpointing requested but SegResNet layers not found — skipping[/yellow]")
        return model

    def ckpt_encode(self, x):
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []
        for down in self.down_layers:
            x = checkpoint(down, x, use_reentrant=False)
            down_x.append(x)
        return x, down_x

    def ckpt_decode(self, x, down_x):
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = checkpoint(upl, x, use_reentrant=False)
        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    model.encode = types.MethodType(ckpt_encode, model)
    model.decode = types.MethodType(ckpt_decode, model)
    console.print("[green]Gradient checkpointing enabled on SegResNet down/up stages[/green]")
    return model


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
            n_blocks_per_stage=_expand_blocks(cfg.get("n_blocks_encoder", 2), n_stages, "n_blocks_encoder"),
            n_conv_per_stage_decoder=_expand_blocks(cfg.get("n_blocks_decoder", 2), n_stages - 1, "n_blocks_decoder"),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs=norm_kwargs,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=cfg.get("deep_supervision", True),
        )
        console.print(
            f"[green]Created nnU-Net v2 ResidualEncoderUNet with features={features_per_stage}, "
            f"encoder blocks={_expand_blocks(cfg.get('n_blocks_encoder', 2), n_stages, 'n_blocks_encoder')}[/green]"
        )
    else:
        model = PlainConvUNet(
            input_channels=in_ch,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv3d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            num_classes=out_ch,
            n_conv_per_stage=_expand_blocks(cfg.get("n_blocks_encoder", 2), n_stages, "n_blocks_encoder"),
            n_conv_per_stage_decoder=_expand_blocks(cfg.get("n_blocks_decoder", 2), n_stages - 1, "n_blocks_decoder"),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs=norm_kwargs,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=cfg.get("deep_supervision", True),
        )
        console.print(f"[green]Created nnU-Net v2 PlainConvUNet with features={features_per_stage}[/green]")

    if cfg.get("grad_checkpointing", False):
        model = _enable_grad_checkpointing(model)

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

    # Accept the native-pipeline bridge tag "nnunet_v2_native_resenc" as well, so the
    # same config is runnable through this train.py loop (e.g. the overfit sanity check).
    if name.startswith("nnunet_v2"):
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
        if cfg.get("grad_checkpointing", False):
            model = _enable_segresnet_grad_checkpointing(model)

    else:
        raise ValueError(f"Unknown model: {name}. Choose from: nnunet_v2, dynunet, swin_unetr, segresnet")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[dim]Trainable parameters: {n_params:,}[/dim]")

    return model
