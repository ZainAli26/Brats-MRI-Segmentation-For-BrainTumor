"""exp25 as native nnU-Net: MONAI SegResNetDS instead of the planned ResEnc U-Net.

exp25 is exp20 with the network swapped to SegResNetDS (Myronenko's BraTS-winner family,
the MONAI Auto3DSeg default) AND its optimizer swapped to the one that family is trained
with — AdamW lr 2e-4, weight decay 1e-5 (poly schedule and everything else kept). Data,
plans-derived patch/batch, splits, loss (Dice+CE) and the 750-epoch budget stay exp20's:

    exp20:  ResEnc U-Net 102.4M   (nnUNetTrainer_750epochs,  -p nnUNetResEncUNetPlans_11G)
    exp24:  shallow ResEnc 31.4M  (nnUNetTrainer_750epochs,  -p nnUNetResEncShallow_11G)
    exp25:  SegResNetDS   87.2M   (this trainer,             -p nnUNetResEncUNetPlans_11G)

The plan's architecture entry is deliberately ignored — the plan still supplies patch size
[128, 192, 128], batch size 2, the preprocessed data identifier (nnUNetPlans_3d_fullres,
shared cache, zero re-preprocessing) and the fold splits. nnU-Net keys the results
directory off the trainer name, so this coexists with exp20/exp22/exp24 under Dataset104.

Why not exp20's SGD: the first attempt (2026-08-25, fold 0) inherited nnU-Net's
SGD lr 0.01 / momentum 0.99 / nesterov and DIVERGED — train loss climbed monotonically
from 0.33 (epoch 1) to ~8-9 by epoch ~15 and never recovered through epoch 46+ (run
preserved at /workspace/diverged_runs/exp25_fold0_sgd_diverged). That recipe is tuned
for nnU-Net's own architectures; MONAI/Auto3DSeg trains SegResNetDS with AdamW ~2e-4,
which is what configure_optimizers uses below. The comparison framing is therefore
"each architecture under a recipe that works for it", not a pure single-variable swap.
AdamW's second moment buffer adds ~350 MB optimizer state over SGD (87M params) —
still well inside the 11 G budget next to the measured 8.15 GiB SGD peak.

Config: init_filters=32, blocks_down=(1,2,2,4,4), instance norm, relu, deconv upsampling,
dsdepth=4 — 87,168,244 params. Five resolution stages (vs the plan's six): a sixth
SegResNetDS stage would put 1024 channels at the bottleneck (~226M params in that stage
alone, the family doubles filters per stage with no cap). Measured on the 5090: 8.15 GiB
peak (autocast fwd+bwd+SGD, deep supervision, batch 2, full patch) — inside the 11 G
budget without gradient checkpointing.

Deep supervision wiring (the part that actually needs care):
  - MONAI orders DS outputs full-res first — the same order nnU-Net's weight ladder
    expects — but returns the list only in train() mode. nnU-Net's per-epoch
    validation_step runs the network in eval() and still consumes the list, so
    SegResNetDSFlagged keys the list-vs-single decision off an explicit
    .deep_supervision flag instead of .training.
  - _get_deep_supervision_scales is overridden to the four scales this net emits
    (1, 1/2, 1/4, 1/8); the base class would derive five from the plan's six stages and
    the target/output lists would disagree. The base loss zeroes the lowest weight, so
    effective supervision is at 1, 1/2, 1/4 — one dead lowest head, exactly nnU-Net's own
    convention.
  - dsdepth is ALWAYS 4, even when built for inference: state-dict keys then match
    between training and prediction builds (dsdepth=1 would drop the extra head convs and
    make load_state_dict fail strict loading).

INSTALL: nnU-Net resolves trainer names from its own package tree, so copy this file in
before `-tr nnUNetTrainerSegResNetDS_750epochs` resolves:

    cp nnunet_native/nnUNetTrainerSegResNetDS.py \
       "$(python3 -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')"/training/nnUNetTrainer/variants/network_architecture/

A pod rebuild or nnunetv2 reinstall silently removes the copy — this repo file is the
source of truth; the training driver re-copies idempotently before every exp25 run.
"""
import torch

from monai.networks.nets import SegResNetDS
from torch._dynamo import OptimizedModule

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_750epochs,
)


class SegResNetDSFlagged(SegResNetDS):
    """SegResNetDS whose DS-list-vs-single-output choice is an explicit flag.

    MONAI's _forward returns the deep-supervision list only while self.training is True;
    nnU-Net needs the list in eval() too (validation_step) and a single tensor during
    sliding-window prediction regardless of mode. The body below is MONAI 1.6.0's
    _forward verbatim except for the final return condition.
    """

    deep_supervision: bool = True

    def _forward(self, x: torch.Tensor):
        if self.preprocess is not None:
            x = self.preprocess(x)

        if not self.is_valid_shape(x):
            raise ValueError(
                f"Input spatial dims {x.shape} must be divisible by {self.shape_factor()}"
            )

        x_down = self.encoder(x)

        x_down.reverse()
        x = x_down.pop(0)

        if len(x_down) == 0:
            x_down = [torch.zeros(1, device=x.device, dtype=x.dtype)]

        outputs: list[torch.Tensor] = []

        i = 0
        for level in self.up_layers:
            x = level["upsample"](x)
            x += x_down.pop(0)
            x = level["blocks"](x)

            if len(self.up_layers) - i <= self.dsdepth:
                outputs.append(level["head"](x))
            i = i + 1

        outputs.reverse()

        if not self.deep_supervision or len(outputs) == 1:
            return outputs[0]

        return outputs


class nnUNetTrainerSegResNetDS_750epochs(nnUNetTrainer_750epochs):
    """nnUNetTrainer_750epochs with SegResNetDS (87.2M) and its native AdamW recipe.

    SGD 0.01/0.99 diverged on this net (see module docstring); AdamW lr 2e-4 wd 1e-5 is
    the MONAI Auto3DSeg recipe for SegResNetDS. The poly LR schedule is kept.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 2e-4
        self.weight_decay = 1e-5

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr,
                                      weight_decay=self.weight_decay)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        # The plan's ResEnc architecture kwargs are ignored on purpose (see module
        # docstring); dsdepth stays 4 in every build for state-dict parity.
        net = SegResNetDSFlagged(
            spatial_dims=3,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            init_filters=32,
            blocks_down=(1, 2, 2, 4, 4),
            dsdepth=4,
            norm="instance",
            act="relu",
            upsample_mode="deconv",
        )
        net.deep_supervision = enable_deep_supervision
        return net

    def _get_deep_supervision_scales(self):
        # Four DS outputs at 1, 1/2, 1/4, 1/8 (five stages), not the base class's five
        # derived from the plan's six-stage pooling.
        if self.enable_deep_supervision:
            return [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]]
        return None

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.deep_supervision = enabled
