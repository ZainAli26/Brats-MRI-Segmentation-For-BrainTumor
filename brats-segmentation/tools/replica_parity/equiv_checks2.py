#!/usr/bin/env python3
"""Step-level equivalence: nnUNetTrainer.train_step / validation_step / inference vs the replica.

check_plan..check_preprocessing (equiv_checks.py) prove the *ingredients* match. These three
prove the *optimisation step* matches: given identical weights and an identical batch, the
two trainers must produce the same loss, the same gradient after clipping, and the same
updated weights — which is the thing a 2-epoch loss curve can only hint at.

    PYTHONPATH=<nnUNet242> python3 equiv_checks2.py [train_step|val_step|inference]
"""
from __future__ import annotations

import json
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
CACHE = SP / "replica_cache/Dataset500_ReplicaCmp_fp32"
NATIVE_RUN = (REPO / "nnunet_data/nnUNet_results/Dataset500_ReplicaCmp"
              / "nnUNetTrainer_2epochs__nnUNetResEncUNetMPlans__3d_fullres" / "fold_0")
NUM_CLASSES, NUM_CHANNELS = 5, 4

sys.path.insert(0, str(REPO))
RESULTS = []

# torch >= 2.6 flipped torch.load's default to weights_only=True; nnU-Net 2.4.2 predates that
# and calls torch.load bare when loading a checkpoint. Compat shim only -- it changes nothing
# about what is computed.
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": k.get("weights_only", False)})

# Determinism: cuDNN autotuning would otherwise pick different algorithms for the two runs
# and the comparison would be dominated by kernel choice rather than by the recipe.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       {detail}" if detail else ""))


# ------------------------------------------------------------------ fixtures
def load_json(p):
    with open(p) as f:
        return json.load(f)


def native_trainer():
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    tr = nnUNetTrainer(load_json(PLANS), "3d_fullres", 0, load_json(PREP / "dataset.json"),
                       unpack_dataset=False, device=torch.device("cuda"))
    tr.initialize()
    tr.set_deep_supervision_enabled(True)
    return tr


def replica_trainer():
    from src.nnunet_replica.config import ReplicaConfig
    from src.nnunet_replica.trainer import NNUNetReplicaTrainer
    config = {
        "data": {"modalities": ["t1c", "t1n", "t2f", "t2w"], "num_classes": NUM_CLASSES},
        "evaluation": {"regions": {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3], "RC": [4]}},
        "replica": {"plans_json": str(PLANS), "preprocessed_dir": str(CACHE),
                    "num_epochs": 2, "num_workers": 0},
    }
    rcfg = ReplicaConfig.from_config(config)
    tr = NNUNetReplicaTrainer(config, rcfg, 0, [], [], SP / "runs/_equiv_scratch")

    def _no_loaders():
        # Everything _build_dataloaders sets *except* the loaders themselves: a fixed batch
        # is supplied by hand, and spawning 12 workers here would be pure overhead.
        from src.nnunet_replica.augmentation import configure_rotation_and_initial_patch_size
        _, _, initial, mirror = configure_rotation_and_initial_patch_size(tr.patch_size)
        tr.initial_patch_size, tr.mirror_axes = initial, mirror
    tr._build_dataloaders = _no_loaders
    tr.initialize()
    return tr


def fixed_batch(patch_size, ds_scales, batch_size=2, seed=17, train=True):
    """One realistic augmented batch, produced deterministically by the replica's own pipeline."""
    from src.nnunet_replica.augmentation import (
        configure_rotation_and_initial_patch_size, get_training_transforms,
        get_validation_transforms, mask_channels_for_norm,
    )
    from src.nnunet_replica.dataloading import PatchSampler3D, PreprocessedDataset

    ids = load_json(PREP / "splits_final.json")[0]["train"][:20]
    ds = PreprocessedDataset(CACHE, ids)
    rot, dummy, initial, mirror = configure_rotation_and_initial_patch_size(patch_size)
    sampled = initial if train else patch_size
    sampler = PatchSampler3D(ds, batch_size, sampled, patch_size, [1, 2, 3, 4], 0.33,
                             rng=np.random.RandomState(seed))
    raw = sampler.generate_batch()
    tf = (get_training_transforms(patch_size, rot, ds_scales, mirror, dummy,
                                  use_mask_for_norm=mask_channels_for_norm([True] * 4, NUM_CHANNELS))
          if train else get_validation_transforms(ds_scales))
    np.random.seed(seed)
    out = tf(**{"data": raw["data"], "seg": raw["seg"]})
    return {"data": out["data"], "target": out["target"], "keys": raw["keys"]}


def sync_weights(dst_net, src_net):
    dst_net.load_state_dict(src_net.state_dict())


# -------------------------------------------------------------------- checks
def _run_steps(trainer, init_state, batches, is_native):
    """Three optimiser steps from a fixed start on fixed batches; return losses + end weights."""
    net = trainer.network
    net.load_state_dict(init_state)
    trainer.optimizer = torch.optim.SGD(net.parameters(), 1e-2, weight_decay=3e-5,
                                        momentum=0.99, nesterov=True)
    trainer.grad_scaler = torch.amp.GradScaler("cuda")
    net.train()
    losses = []
    for b in batches:
        out = trainer.train_step({"data": b["data"].clone(),
                                  "target": [t.clone() for t in b["target"]]})
        losses.append(float(out["loss"]))
    end_state = {k: v.detach().to("cpu").clone() for k, v in net.state_dict().items()}
    return losses, end_state


def check_train_step():
    """Same weights, same batches, three SGD steps -- run sequentially, because two
    102 M-parameter networks with live backward graphs do not fit on an 8 GB card."""
    patch = None
    scales = None

    nat = native_trainer()
    patch = list(nat.configuration_manager.patch_size)
    # deep_supervision_scales from the native side, so the batch targets are built once
    scales = nat._get_deep_supervision_scales()
    init_state = {k: v.detach().to("cpu").clone() for k, v in nat.network.state_dict().items()}
    batches = [fixed_batch(patch, scales, nat.batch_size, seed=100 + i, train=True)
               for i in range(3)]
    ln, wn = _run_steps(nat, init_state, batches, True)
    del nat
    torch.cuda.empty_cache()

    rep = replica_trainer()
    lr_, wr = _run_steps(rep, init_state, batches, False)
    del rep
    torch.cuda.empty_cache()

    rows = []
    worst_loss = max(abs(a - b) for a, b in zip(ln, lr_))
    worst_w = max(float((wn[k].float() - wr[k].float()).abs().max()) for k in wn)
    for i, (a, b) in enumerate(zip(ln, lr_)):
        rows.append(f"step {i}: native loss {a:.6f}  replica loss {b:.6f}  |dloss|={abs(a-b):.2e}")
    rows.append(f"after 3 steps: max |weight difference| = {worst_w:.3e} "
                f"(over {len(wn)} tensors, {sum(v.numel() for v in wn.values()):,} values)")
    ok = worst_loss < 1e-4 and worst_w < 1e-5
    report("train_step (loss, AMP, grad clip 12, SGD+Nesterov update)", ok, "\n       ".join(rows))


def check_val_step():
    nat = native_trainer()
    rep = replica_trainer()
    patch = list(nat.configuration_manager.patch_size)
    scales = rep.deep_supervision_scales
    sync_weights(rep.network, nat.network)
    nat.network.eval(); rep.network.eval()

    b = fixed_batch(patch, scales, nat.batch_size, seed=55, train=False)
    on = nat.validation_step({"data": b["data"].clone(), "target": [t.clone() for t in b["target"]]})
    orr = rep.validation_step({"data": b["data"].clone(), "target": [t.clone() for t in b["target"]]})

    dl = abs(float(on["loss"]) - float(orr["loss"]))
    dtp = float(np.abs(np.asarray(on["tp_hard"]) - np.asarray(orr["tp_hard"])).max())
    dfp = float(np.abs(np.asarray(on["fp_hard"]) - np.asarray(orr["fp_hard"])).max())
    dfn = float(np.abs(np.asarray(on["fn_hard"]) - np.asarray(orr["fn_hard"])).max())
    ok = dl < 2e-3 and dtp == 0 and dfp == 0 and dfn == 0
    report("validation_step (loss + hard TP/FP/FN for pseudo-Dice)", ok,
           f"loss native {float(on['loss']):.6f} vs replica {float(orr['loss']):.6f} (|d|={dl:.2e}); "
           f"TP/FP/FN max|diff| = {dtp:.0f}/{dfp:.0f}/{dfn:.0f}; "
           f"TP native {np.asarray(on['tp_hard']).astype(int).tolist()}")
    del nat, rep
    torch.cuda.empty_cache()


def check_inference():
    """Native nnUNetPredictor vs replica predict_sliding_window, same weights, same volume."""
    import pickle
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from src.nnunet_replica.inference import predict_sliding_window
    from src.nnunet_replica.network import build_network
    from src.nnunet_replica.plans import Plans

    ckpt = NATIVE_RUN / "checkpoint_final.pth"
    if not ckpt.is_file():
        report("inference (sliding window + Gaussian + mirroring TTA)", False,
               f"native checkpoint not found yet: {ckpt}")
        return

    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           perform_everything_on_device=True, device=torch.device("cuda"),
                           verbose=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(str(NATIVE_RUN.parent), use_folds=(0,),
                                              checkpoint_name="checkpoint_final.pth")
    patch = list(pred.configuration_manager.patch_size)

    plan_cfg = Plans.load(PLANS).get_configuration("3d_fullres")
    rep_net = build_network(plan_cfg, NUM_CHANNELS, NUM_CLASSES, deep_supervision=False).cuda()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["network_weights"]
    rep_net.load_state_dict(state)
    rep_net.eval()

    val_ids = load_json(PREP / "splits_final.json")[0]["val"][:3]
    rows, worst_agree, worst_dice = [], 1.0, 1.0
    for case in val_ids:
        data = np.load(CACHE / f"{case}.npy").astype(np.float32)
        seg = np.load(CACHE / f"{case}_seg.npy")
        props = pickle.load(open(CACHE / f"{case}.pkl", "rb"))

        pred.network.load_state_dict(state)
        logits_n = pred.predict_sliding_window_return_logits(torch.from_numpy(data))
        seg_n = logits_n.float().argmax(0).cpu().numpy().astype(np.uint8)

        logits_r = predict_sliding_window(rep_net, torch.from_numpy(data), patch, NUM_CLASSES,
                                          tile_step_size=0.5, use_gaussian=True,
                                          mirror_axes=(0, 1, 2), device=torch.device("cuda"),
                                          amp=True)
        seg_r = logits_r.argmax(0).cpu().numpy().astype(np.uint8)

        agree = float((seg_n == seg_r).mean())
        gt = np.where(np.asarray(seg[0]) < 0, 0, np.asarray(seg[0]))
        per_class = {}
        for c in range(1, NUM_CLASSES):
            a, b2 = seg_n == c, seg_r == c
            den = a.sum() + b2.sum()
            per_class[c] = float(2 * (a & b2).sum() / den) if den else float("nan")
            # only count classes both predict at all
            if den > 0:
                worst_dice = min(worst_dice, per_class[c])
        # Dice of each prediction against ground truth, to show they score the same.
        def dice_vs_gt(p, labels):
            pm, gm = np.isin(p, labels), np.isin(gt, labels)
            den = pm.sum() + gm.sum()
            return float(2 * (pm & gm).sum() / den) if den else float("nan")
        wt_n, wt_r = dice_vs_gt(seg_n, [1, 2, 3]), dice_vs_gt(seg_r, [1, 2, 3])
        worst_agree = min(worst_agree, agree)
        rows.append(f"{case}: voxel agreement {agree*100:.4f}%  "
                    f"pred-vs-pred Dice per class "
                    f"{ {k: (None if np.isnan(v) else round(v,5)) for k,v in per_class.items()} }  "
                    f"WT Dice vs GT: native {wt_n:.5f} / replica {wt_r:.5f}")

    ok = worst_agree > 0.999 and worst_dice > 0.99
    report("inference (sliding window + Gaussian + mirroring TTA)", ok, "\n       ".join(rows))


CHECKS = {"train_step": check_train_step, "val_step": check_val_step, "inference": check_inference}


def main():
    wanted = sys.argv[1:] or list(CHECKS)
    for name in wanted:
        try:
            CHECKS[name]()
        except Exception as e:
            import traceback
            report(name, False, f"EXCEPTION {type(e).__name__}: {e}\n"
                   + "".join(traceback.format_exc().splitlines(True)[-8:]))
    print("\n" + "=" * 78)
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"{n_ok}/{len(RESULTS)} checks passed")
    out = SP / "equiv_results2.json"
    prev = json.load(open(out)) if out.is_file() else []
    keep = [r for r in prev if r["check"] not in {n for n, _, _ in RESULTS}]
    json.dump(keep + [{"check": n, "pass": bool(o), "detail": d} for n, o, d in RESULTS],
              open(out, "w"), indent=1)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

