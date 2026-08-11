#!/usr/bin/env python3
"""Side-by-side of the native nnU-Net run and the replica run on Dataset500 fold 0."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/brats-segmentation")
# Durable work dir. This lived in the session scratchpad once; a reboot destroyed it
# along with the cache and every run. nnunet_data/ is gitignored, so heavy artefacts
# stay out of git while these scripts stay in it.
SP = Path("/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/"
          "brats-segmentation/nnunet_data/replica_parity")
NATIVE = (REPO / "nnunet_data/nnUNet_results/Dataset500_ReplicaCmp"
          / "nnUNetTrainer_2epochs__nnUNetResEncUNetMPlans__3d_fullres" / "fold_0")
CLASS_NAMES = ["NETC", "SNFH", "ET", "RC"]


def native_logging():
    ck = NATIVE / "checkpoint_final.pth"
    if ck.is_file():
        lg = torch.load(ck, map_location="cpu", weights_only=False)["logging"]
        out = {k: list(v) for k, v in lg.items()}
        # nnUNetLogger calls it dice_per_class_or_region; the replica calls it dice_per_class.
        if "dice_per_class_or_region" in out:
            out["dice_per_class"] = out.pop("dice_per_class_or_region")
        return out
    # Fall back to the text log if the run has not finished.
    logs = sorted(NATIVE.glob("training_log_*.txt"))
    txt = logs[-1].read_text() if logs else ""
    out = {"train_losses": [], "val_losses": [], "dice_per_class": [], "lrs": [],
           "epoch_start_timestamps": [], "epoch_end_timestamps": []}
    for m in re.finditer(r"train_loss (-?[\d.]+)", txt):
        out["train_losses"].append(float(m.group(1)))
    for m in re.finditer(r"val_loss (-?[\d.]+)", txt):
        out["val_losses"].append(float(m.group(1)))
    for m in re.finditer(r"Pseudo dice \[([^\]]*)\]", txt):
        out["dice_per_class"].append([float(x) for x in m.group(1).replace("nan", "nan").split(",")])
    for m in re.finditer(r"Current learning rate: ([\d.e+-]+)", txt):
        out["lrs"].append(float(m.group(1)))
    for m in re.finditer(r"Epoch time: ([\d.]+) s", txt):
        out.setdefault("epoch_times", []).append(float(m.group(1)))
    return out


def replica_logging(tag="replica_cmp"):
    p = SP / "runs" / tag / "cmp_summary.json"
    if p.is_file():
        return json.load(open(p))["logging"]
    p2 = SP / "runs" / tag / "training_history.json"
    return json.load(open(p2))


def epoch_times(lg):
    s, e = lg.get("epoch_start_timestamps") or [], lg.get("epoch_end_timestamps") or []
    if s and e and len(s) == len(e):
        return [round(b - a, 1) for a, b in zip(s, e)]
    return lg.get("epoch_times", [])


def mean_fg(dpc):
    return float(np.nanmean(dpc)) if dpc is not None else float("nan")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "replica_cmp"
    n, r = native_logging(), replica_logging(tag)
    nt, rt = epoch_times(n), epoch_times(r)
    n_ep = min(len(n["train_losses"]), len(r["train_losses"]))

    W = 96
    print("=" * W)
    print("nnU-Net native v2.4.2   vs   replica loop  —  Dataset500_ReplicaCmp, fold 0")
    print("nnUNetResEncUNetMPlans (8 GB ResEnc-M) / 3d_fullres  •  161 train / 40 val cases")
    print("=" * W)
    hdr = f"{'metric':<28}{'native':>16}{'replica':>16}{'difference':>16}"
    for ep in range(n_ep):
        print(f"\n--- epoch {ep} " + "-" * (W - 12))
        print(hdr)
        rows = [
            ("learning rate", n["lrs"][ep], r["lrs"][ep]),
            ("train loss", n["train_losses"][ep], r["train_losses"][ep]),
            ("val loss", n["val_losses"][ep], r["val_losses"][ep]),
        ]
        for name, a, b in rows:
            print(f"{name:<28}{a:>16.5f}{b:>16.5f}{b - a:>+16.5f}")
        dn, dr = n["dice_per_class"][ep], r["dice_per_class"][ep]
        for i, cn in enumerate(CLASS_NAMES):
            a, b = float(dn[i]), float(dr[i])
            print(f"{'pseudo Dice ' + cn:<28}{a:>16.5f}{b:>16.5f}{b - a:>+16.5f}")
        a, b = mean_fg(dn), mean_fg(dr)
        print(f"{'mean fg pseudo Dice':<28}{a:>16.5f}{b:>16.5f}{b - a:>+16.5f}")
        if ep < len(nt) and ep < len(rt):
            print(f"{'epoch time (s)':<28}{nt[ep]:>16.1f}{rt[ep]:>16.1f}{rt[ep] - nt[ep]:>+16.1f}")

    print("\n" + "=" * W)
    print("summary over the run")
    print(hdr)
    for label, key in [("final train loss", "train_losses"), ("final val loss", "val_losses")]:
        a, b = n[key][n_ep - 1], r[key][n_ep - 1]
        print(f"{label:<28}{a:>16.5f}{b:>16.5f}{b - a:>+16.5f}")
    a = mean_fg(n["dice_per_class"][n_ep - 1]); b = mean_fg(r["dice_per_class"][n_ep - 1])
    print(f"{'final mean fg pseudo Dice':<28}{a:>16.5f}{b:>16.5f}{b - a:>+16.5f}")
    da = n["train_losses"][0] - n["train_losses"][n_ep - 1]
    db = r["train_losses"][0] - r["train_losses"][n_ep - 1]
    print(f"{'train-loss drop e0->e' + str(n_ep - 1):<28}{da:>16.5f}{db:>16.5f}{db - da:>+16.5f}")
    if nt and rt:
        print(f"{'mean epoch time (s)':<28}{np.mean(nt[:n_ep]):>16.1f}{np.mean(rt[:n_ep]):>16.1f}"
              f"{np.mean(rt[:n_ep]) - np.mean(nt[:n_ep]):>+16.1f}")
    print("=" * W)

    json.dump({"native": n, "replica": r, "native_epoch_times": nt, "replica_epoch_times": rt},
              open(SP / "run_comparison.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

