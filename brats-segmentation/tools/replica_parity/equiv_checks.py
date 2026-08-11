#!/usr/bin/env python3
"""Component-by-component equivalence: our replica loop vs native nnU-Net v2.4.2.

A 1-2 epoch loss curve is a weak test — SGD noise at lr 1e-2 on 250 steps easily hides a
wrong augmentation probability or a permuted deep-supervision weight. These checks compare
the two implementations directly, piece by piece, on identical inputs, so the end-to-end
run has something to be interpreted against.

Run from brats-segmentation/ with PYTHONPATH pointing at the 2.4.2 source tree:
    PYTHONPATH=<nnUNet242> python3 equiv_checks.py [check ...]
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

REPO = Path("/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/brats-segmentation")
# Durable work dir. This lived in the session scratchpad once; a reboot destroyed it
# along with the cache and every run. nnunet_data/ is gitignored, so heavy artefacts
# stay out of git while these scripts stay in it.
SP = Path("/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/"
          "brats-segmentation/nnunet_data/replica_parity")
PLANS = REPO / "plans/nnUNetResEncUNetMPlans_8G_cmp.json"
PREP = REPO / "nnunet_data/nnUNet_preprocessed/Dataset500_ReplicaCmp"
RAW = REPO / "nnunet_data/nnUNet_raw/Dataset500_ReplicaCmp"
NUM_CLASSES = 5
NUM_CHANNELS = 4

sys.path.insert(0, str(REPO))

RESULTS = []


def check(name):
    def deco(fn):
        fn._check_name = name
        return fn
    return deco


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       {detail}" if detail else ""))


def close(a, b, tol=0.0):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    d = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    return d <= tol, f"max|diff| = {d:.3e}"


# ---------------------------------------------------------------- shared fixtures
def native_managers():
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    pm = PlansManager(str(PLANS))
    cm = pm.get_configuration("3d_fullres")
    dataset_json = json.load(open(PREP / "dataset.json"))
    return pm, cm, dataset_json


def replica_plan():
    from src.nnunet_replica.plans import Plans
    return Plans.load(PLANS).get_configuration("3d_fullres")


# ---------------------------------------------------------------------- checks
@check("plan parsing (patch / batch / batch_dice / norm / resampling order)")
def check_plan():
    _, cm, _ = native_managers()
    r = replica_plan()
    fields = {
        "patch_size": (list(cm.patch_size), list(r.patch_size)),
        "batch_size": (cm.batch_size, r.batch_size),
        "batch_dice": (cm.batch_dice, r.batch_dice),
        "spacing": (list(cm.spacing), list(r.spacing)),
        "normalization_schemes": (list(cm.normalization_schemes), list(r.normalization_schemes)),
        "use_mask_for_norm": (list(cm.use_mask_for_norm), list(r.use_mask_for_norm)),
        "order_data": (cm.configuration["resampling_fn_data_kwargs"]["order"], r.resampling_order_data),
        "order_seg": (cm.configuration["resampling_fn_seg_kwargs"]["order"], r.resampling_order_seg),
        "pool_op_kernel_sizes": ([list(x) for x in cm.pool_op_kernel_sizes],
                                 [list(x) for x in r.pool_op_kernel_sizes]),
    }
    bad = {k: v for k, v in fields.items() if v[0] != v[1]}
    report("plan parsing", not bad, json.dumps(bad) if bad else
           f"patch {list(cm.patch_size)}  batch {cm.batch_size}  batch_dice {cm.batch_dice}")


@check("deep-supervision scales and loss weights")
def check_ds():
    _, cm, _ = native_managers()
    r = replica_plan()
    native_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(cm.pool_op_kernel_sizes), axis=0))[:-1]
    ok1, d1 = close(native_scales, r.deep_supervision_scales)

    nw = np.array([1 / (2 ** i) for i in range(len(native_scales))], dtype=np.float64)
    nw[-1] = 0
    nw = nw / nw.sum()
    from src.nnunet_replica.loss import deep_supervision_weights
    rw = deep_supervision_weights(len(r.deep_supervision_scales), zero_lowest=True)
    ok2, d2 = close(nw, rw)
    report("deep supervision scales + weights", ok1 and ok2,
           f"scales {d1}; weights {d2}; weights={[round(float(x),4) for x in rw]}")


@check("network: architecture and initialisation, bitwise")
def check_network():
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from src.nnunet_replica.network import build_network, count_parameters
    pm, cm, dj = native_managers()

    torch.manual_seed(1234)
    net_native = get_network_from_plans(
        cm.network_arch_class_name, cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import, NUM_CHANNELS, NUM_CLASSES,
        allow_init=True, deep_supervision=True,
    )
    torch.manual_seed(1234)
    net_replica = build_network(replica_plan(), NUM_CHANNELS, NUM_CLASSES,
                               deep_supervision=True, grad_checkpointing=False)

    sd_n, sd_r = net_native.state_dict(), net_replica.state_dict()
    keys_ok = list(sd_n.keys()) == list(sd_r.keys())
    n_par = count_parameters(net_native)
    r_par = count_parameters(net_replica)

    maxdiff = 0.0
    if keys_ok:
        for k in sd_n:
            if sd_n[k].shape != sd_r[k].shape:
                maxdiff = float("inf")
                break
            maxdiff = max(maxdiff, float((sd_n[k].float() - sd_r[k].float()).abs().max()))

    # Forward equality on a fixed input (native weights copied into the replica net so this
    # isolates the graph from the init).
    #
    # nnUNetTrainer builds the network with deep supervision OFF -- get_network_from_plans
    # copies arch_kwargs *before* injecting the flag, so it never reaches the constructor --
    # and turns it on in on_train_start() via set_deep_supervision_enabled(). Replicate that
    # here so both nets emit the same number of heads.
    net_native.decoder.deep_supervision = True
    net_replica.load_state_dict(sd_n)
    x = torch.randn(1, NUM_CHANNELS, *cm.patch_size, generator=torch.Generator().manual_seed(7))
    net_native.eval(); net_replica.eval()
    with torch.no_grad():
        on, orr = net_native(x), net_replica(x)
    fwd = max(float((a - b).abs().max()) for a, b in zip(on, orr))
    heads_ok = len(on) == len(orr)

    ok = keys_ok and n_par == r_par and maxdiff == 0.0 and fwd == 0.0 and heads_ok
    report("network architecture + init (bitwise)", ok,
           f"params {n_par:,} vs {r_par:,}; state_dict keys identical={keys_ok}; "
           f"init max|diff|={maxdiff:.3e}; forward max|diff|={fwd:.3e}; heads {len(on)} vs {len(orr)}")


@check("loss: Dice+CE under deep supervision")
def check_loss():
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from src.nnunet_replica.loss import build_loss

    _, cm, _ = native_managers()
    r = replica_plan()
    scales = r.deep_supervision_scales

    native = DC_and_CE_loss({"batch_dice": cm.batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": False},
                            {}, weight_ce=1, weight_dice=1, ignore_label=None,
                            dice_class=MemoryEfficientSoftDiceLoss)
    w = np.array([1 / (2 ** i) for i in range(len(scales))], dtype=np.float64)
    w[-1] = 0
    w = w / w.sum()
    native = DeepSupervisionWrapper(native, w)
    rep = build_loss(batch_dice=cm.batch_dice, deep_supervision_scales=scales,
                     loss_name="dice_ce", zero_lowest_ds=True)

    g = torch.Generator().manual_seed(11)
    patch = list(cm.patch_size)
    outputs, targets = [], []
    for s in scales:
        shp = [max(1, int(round(p * f))) for p, f in zip(patch, s)]
        outputs.append(torch.randn(2, NUM_CLASSES, *shp, generator=g))
        targets.append(torch.randint(0, NUM_CLASSES, (2, 1, *shp), generator=g).float())

    ln = float(native(outputs, targets))
    lr = float(rep(outputs, targets))
    # Also check the un-wrapped single-head loss.
    ln1 = float(DC_and_CE_loss({"batch_dice": cm.batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": False},
                               {}, weight_ce=1, weight_dice=1, ignore_label=None,
                               dice_class=MemoryEfficientSoftDiceLoss)(outputs[0], targets[0]))
    lr1 = float(getattr(rep, "loss")(outputs[0], targets[0]))
    ok = abs(ln - lr) < 1e-6 and abs(ln1 - lr1) < 1e-6
    report("loss Dice+CE (+deep supervision)", ok,
           f"DS-wrapped: native {ln:.8f} vs replica {lr:.8f} (|d|={abs(ln-lr):.2e}); "
           f"single head: {ln1:.8f} vs {lr1:.8f} (|d|={abs(ln1-lr1):.2e})")


@check("poly LR schedule")
def check_lr():
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler as NativeLR
    from src.nnunet_replica.lr_scheduler import PolyLRScheduler as RepLR
    net = torch.nn.Linear(2, 2)
    on = torch.optim.SGD(net.parameters(), 1e-2)
    orr = torch.optim.SGD(net.parameters(), 1e-2)
    sn, sr = NativeLR(on, 1e-2, 1000), RepLR(orr, 1e-2, 1000)
    a, b = [], []
    for e in [0, 1, 2, 50, 250, 500, 999]:
        sn.step(e); sr.step(e)
        a.append(on.param_groups[0]["lr"]); b.append(orr.param_groups[0]["lr"])
    ok, d = close(a, b, 0.0)
    report("poly LR schedule", ok, f"{d}; lr(0)={a[0]:.6f} lr(1)={a[1]:.6f} lr(999)={a[-1]:.3e}")


@check("rotation / mirror axes / rotation-inflated sampling patch")
def check_rotation():
    from src.nnunet_replica.augmentation import configure_rotation_and_initial_patch_size
    from nnunetv2.configuration import ANISO_THRESHOLD
    from batchgenerators.augmentations.utils import rotate_coords_3d  # noqa: F401
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    _, cm, _ = native_managers()
    patch = list(cm.patch_size)

    # Native: call the real method on a stub carrying just what it reads.
    class Stub:
        configuration_manager = cm
        print_to_log_file = staticmethod(lambda *a, **k: None)
    nat = nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size(Stub())
    n_rot, n_dummy, n_initial, n_mirror = nat

    r_rot, r_dummy, r_initial, r_mirror = configure_rotation_and_initial_patch_size(patch)
    ok = (list(map(int, n_initial)) == list(map(int, r_initial))
          and tuple(n_mirror) == tuple(r_mirror)
          and bool(n_dummy) == bool(r_dummy)
          and all(np.allclose(n_rot[k], r_rot[k]) for k in "xyz"))
    report("rotation config + inflated patch", ok,
           f"initial patch native {list(map(int,n_initial))} vs replica {list(map(int,r_initial))}; "
           f"mirror {tuple(n_mirror)} vs {tuple(r_mirror)}; dummy2d {n_dummy} vs {r_dummy}; "
           f"ANISO_THRESHOLD={ANISO_THRESHOLD}")


@check("augmentation pipeline: transform list and pixel output")
def check_augmentation():
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from src.nnunet_replica.augmentation import (
        configure_rotation_and_initial_patch_size, get_training_transforms,
        get_validation_transforms, mask_channels_for_norm,
    )
    _, cm, _ = native_managers()
    r = replica_plan()
    patch = list(cm.patch_size)
    rot, dummy, initial, mirror = configure_rotation_and_initial_patch_size(patch)
    scales = r.deep_supervision_scales
    umn = mask_channels_for_norm(cm.use_mask_for_norm, NUM_CHANNELS)

    nat = nnUNetTrainer.get_training_transforms(
        patch, rot, scales, mirror, dummy,
        order_resampling_data=3, order_resampling_seg=1, border_val_seg=-1,
        use_mask_for_norm=umn, is_cascaded=False, foreground_labels=None,
        regions=None, ignore_label=None,
    )
    rep = get_training_transforms(patch, rot, scales, mirror, dummy, use_mask_for_norm=umn,
                                  order_resampling_data=3, order_resampling_seg=1)

    # The replica ports nnU-Net's DownsampleSegForDSTransform2 under a shorter name; treat
    # that rename as equal (the pixel-level comparison below is what actually decides).
    alias = {"DownsampleSegForDSTransform": "DownsampleSegForDSTransform2"}
    names_n = [type(t).__name__ for t in nat.transforms]
    names_r = [alias.get(type(t).__name__, type(t).__name__) for t in rep.transforms]
    list_ok = names_n == names_r

    # Parameter comparison for every transform that carries probabilities.
    param_diffs = []
    for tn, tr in zip(nat.transforms, rep.transforms):
        dn = {k: v for k, v in vars(tn).items() if isinstance(v, (int, float, bool, tuple, list, str, type(None)))}
        dr = {k: v for k, v in vars(tr).items() if isinstance(v, (int, float, bool, tuple, list, str, type(None)))}
        for k in set(dn) & set(dr):
            if isinstance(dn[k], (list, tuple)) and isinstance(dr[k], (list, tuple)):
                same = list(dn[k]) == list(dr[k])
            else:
                same = dn[k] == dr[k]
            if not same:
                param_diffs.append(f"{type(tn).__name__}.{k}: {dn[k]!r} vs {dr[k]!r}")

    # Pixel-level: same input, same numpy seed -> identical output.
    rng = np.random.RandomState(3)
    data = rng.randn(2, NUM_CHANNELS, *initial).astype(np.float32)
    seg = rng.randint(-1, NUM_CLASSES, (2, 1, *initial)).astype(np.int16)

    np.random.seed(999)
    on = nat(**{"data": data.copy(), "seg": seg.copy()})
    np.random.seed(999)
    orr = rep(**{"data": data.copy(), "seg": seg.copy()})
    dd = float((on["data"] - orr["data"]).abs().max())
    dt = max(float((a - b).abs().max()) for a, b in zip(on["target"], orr["target"]))
    shapes_ok = [list(t.shape) for t in on["target"]] == [list(t.shape) for t in orr["target"]]

    # Validation pipeline too.
    natv = nnUNetTrainer.get_validation_transforms(scales, is_cascaded=False, foreground_labels=None,
                                                  regions=None, ignore_label=None)
    repv = get_validation_transforms(scales)
    np.random.seed(5); ovn = natv(**{"data": data.copy(), "seg": seg.copy()})
    np.random.seed(5); ovr = repv(**{"data": data.copy(), "seg": seg.copy()})
    dv = float((ovn["data"] - ovr["data"]).abs().max())

    ok = list_ok and not param_diffs and dd == 0.0 and dt == 0.0 and dv == 0.0 and shapes_ok
    detail = (f"{len(names_n)} transforms, list identical={list_ok}; param diffs={len(param_diffs)}; "
              f"train data max|diff|={dd:.3e}, target max|diff|={dt:.3e}; val data max|diff|={dv:.3e}; "
              f"DS target shapes identical={shapes_ok}")
    if param_diffs:
        detail += "\n       " + "\n       ".join(param_diffs[:8])
    if not list_ok:
        detail += f"\n       native: {names_n}\n       replica: {names_r}"
    report("augmentation pipeline", ok, detail)


@check("patch sampler: forced-foreground pattern and bounding boxes")
def check_sampler():
    from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D  # noqa: F401
    from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
    from src.nnunet_replica.dataloading import PatchSampler3D
    from src.nnunet_replica.augmentation import configure_rotation_and_initial_patch_size

    _, cm, _ = native_managers()
    patch = list(cm.patch_size)
    _, _, initial, _ = configure_rotation_and_initial_patch_size(patch)
    fg = [1, 2, 3, 4]

    # A stub carrying only the fields get_bbox reads.
    class NativeBase(nnUNetDataLoaderBase):
        def __init__(self):
            self.patch_size = np.array(initial)
            self.final_patch_size = np.array(patch)
            self.need_to_pad = (np.array(initial) - np.array(patch)).astype(int)
            self.annotated_classes_key = tuple([0] + fg)
            self.has_ignore = False
            self.oversample_foreground_percent = 0.33
            self.batch_size = cm.batch_size

    nb = NativeBase()
    rs = PatchSampler3D.__new__(PatchSampler3D)
    rs.patch_size = np.array(initial, dtype=int)
    rs.final_patch_size = np.array(patch, dtype=int)
    rs.need_to_pad = (np.array(initial) - np.array(patch)).astype(int)
    rs.foreground_labels = fg
    rs.oversample_foreground_percent = 0.33
    rs.batch_size = cm.batch_size

    # Which batch positions are forced foreground?
    pattern_n = [nb._oversample_last_XX_percent(i) for i in range(cm.batch_size)]
    pattern_r = [rs._do_oversample(i) for i in range(cm.batch_size)]

    shape = (140, 170, 133)
    prng = np.random.RandomState(0)
    class_locations = {c: np.concatenate(
        [np.zeros((60, 1), np.int32), prng.randint(0, shape, (60, 3)).astype(np.int32)], axis=1)
        for c in fg}

    mismatch = []
    for trial in range(200):
        for force_fg in (False, True):
            np.random.seed(trial * 7 + int(force_fg))
            bn = nb.get_bbox(shape, force_fg, class_locations, None)
            np.random.seed(trial * 7 + int(force_fg))
            rs.rng = np.random.mtrand._rand   # the replica's RNG *is* the global one here
            br = rs.get_bbox(shape, force_fg, class_locations)
            if [list(bn[0]), list(bn[1])] != [list(br[0]), list(br[1])]:
                mismatch.append((trial, force_fg, bn, br))

    ok = pattern_n == pattern_r and not mismatch
    report("patch sampler (get_bbox + forced-fg pattern)", ok,
           f"forced-fg pattern native {pattern_n} vs replica {pattern_r}; "
           f"400 bbox draws, mismatches={len(mismatch)}"
           + (f"\n       first: {mismatch[0]}" if mismatch else ""))


@check("preprocessing: replica cache vs nnU-Net's own preprocessed case")
def check_preprocessing():
    from src.nnunet_replica.preprocessing import preprocess_case
    pm, cm, dj = native_managers()
    ids = json.load(open(SP / "cmp_splits_fold0.json"))[0]["train"][:3]

    rows, worst = [], 0.0
    for case in ids:
        npz = PREP / "nnUNetPlans_3d_fullres" / f"{case}.npz"
        pkl = PREP / "nnUNetPlans_3d_fullres" / f"{case}.pkl"
        if not npz.is_file():
            report("preprocessing", False, f"native preprocessed file missing: {npz}")
            return
        z = np.load(npz)
        nd, ns = z["data"].astype(np.float64), z["seg"].astype(np.int16)
        nprops = pickle.load(open(pkl, "rb"))

        imgs = [str(RAW / "imagesTr" / f"{case}_{i:04d}.nii.gz") for i in range(NUM_CHANNELS)]
        rd, rsg, rprops = preprocess_case(
            imgs, str(RAW / "labelsTr" / f"{case}.nii.gz"), replica_plan(),
            foreground_labels=[1, 2, 3, 4], label_map=None,
            transpose_forward=pm.transpose_forward,
        )
        rd = rd.astype(np.float64)
        shape_ok = rd.shape == nd.shape and rsg.shape == ns.shape
        dmax = float(np.abs(rd - nd).max()) if shape_ok else float("inf")
        smax = int(np.abs(rsg.astype(np.int32) - ns.astype(np.int32)).max()) if shape_ok else -1
        bbox_ok = ([list(x) for x in nprops["bbox_used_for_cropping"]]
                   == [list(x) for x in rprops["bbox_used_for_cropping"]])
        # class_locations: same per-class counts (the sampled coordinates use an RNG the
        # two implementations seed independently, so compare counts + membership).
        nloc, rloc = nprops["class_locations"], rprops["class_locations"]
        loc_counts = {int(k): (len(nloc.get(k, [])), len(rloc.get(int(k), []))) for k in nloc}
        loc_ok = all(a == b for a, b in loc_counts.values())
        worst = max(worst, dmax)
        rows.append(f"{case}: shape {nd.shape}=={rd.shape}->{shape_ok}, data max|diff|={dmax:.3e}, "
                    f"seg max|diff|={smax}, bbox_match={bbox_ok}, fg-loc counts match={loc_ok} {loc_counts}")

    ok = worst < 1e-4 and all("->True" in r and "seg max|diff|=0" in r and "bbox_match=True" in r
                              and "counts match=True" in r for r in rows)
    report("preprocessing (crop / z-score / fg locations)", ok, "\n       ".join(rows))


CHECKS = [check_plan, check_ds, check_network, check_loss, check_lr, check_rotation,
          check_augmentation, check_sampler, check_preprocessing]


def main():
    wanted = sys.argv[1:]
    for fn in CHECKS:
        if wanted and fn.__name__ not in wanted and fn.__name__.replace("check_", "") not in wanted:
            continue
        try:
            fn()
        except Exception as e:
            import traceback
            report(fn.__name__, False, f"EXCEPTION {type(e).__name__}: {e}\n"
                   + "".join(traceback.format_exc().splitlines(True)[-6:]))
    print("\n" + "=" * 78)
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"{n_ok}/{len(RESULTS)} checks passed")
    for name, ok, _ in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    json.dump([{"check": n, "pass": bool(o), "detail": d} for n, o, d in RESULTS],
              open(SP / "equiv_results.json", "w"), indent=1)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

