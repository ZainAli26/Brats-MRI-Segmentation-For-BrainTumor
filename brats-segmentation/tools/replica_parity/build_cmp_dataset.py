#!/usr/bin/env python3
"""Build Dataset500_ReplicaCmp: a patient-grouped subset of exp20 fold 0.

Both the native nnU-Net run and the replica run consume exactly these cases, in exactly
this channel order, with exactly this fold-0 partition. Images are symlinked, so the raw
dataset costs no disk.

Channel order is the exp20 config's `data.modalities` (t1c, t1n, t2f, t2w) rather than the
older Dataset102 order, so the replica and the native trainer see the identical tensor.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Durable work dir. This lived in the session scratchpad once; a reboot destroyed it
# along with the cache and every run. nnunet_data/ is gitignored, so heavy artefacts
# stay out of git while these scripts stay in it.
SP = Path("/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/"
          "brats-segmentation/nnunet_data/replica_parity")
REPO = Path("/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor")
RAW_ROOT = REPO / "brats-segmentation/nnunet_data/nnUNet_raw"
MODALITIES = ["t1c", "t1n", "t2f", "t2w"]

TARGET_TRAIN_CASES = 160
TARGET_VAL_CASES = 40
SEED = 42


def patient(case_id: str) -> str:
    return case_id.split("-")[2]


def pick(case_ids, target, rng):
    """Choose whole patients until `target` cases are covered."""
    by_patient = {}
    for c in case_ids:
        by_patient.setdefault(patient(c), []).append(c)
    pats = sorted(by_patient)
    rng.shuffle(pats)
    chosen, n = [], 0
    for p in pats:
        if n >= target:
            break
        chosen.extend(sorted(by_patient[p]))
        n += len(by_patient[p])
    return sorted(chosen)


def main():
    splits = pickle.load(open(SP / "exp20_splits.pkl", "rb"))
    tr_all, va_all = splits["folds"][0]
    # split paths are relative to brats-segmentation/ — make them absolute so the
    # symlinks resolve from anywhere.
    base = REPO / "brats-segmentation"
    dir_of = {Path(p).name: (base / p).resolve() for p in splits["all_dirs"]}

    rng = np.random.RandomState(SEED)
    train = pick(tr_all, TARGET_TRAIN_CASES, rng)
    val = pick(va_all, TARGET_VAL_CASES, rng)
    assert not (set(map(patient, train)) & set(map(patient, val))), "patient leak"

    out = RAW_ROOT / "Dataset500_ReplicaCmp"
    (out / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out / "labelsTr").mkdir(parents=True, exist_ok=True)

    for case in train + val:
        src = dir_of[case]
        for i, m in enumerate(MODALITIES):
            link = out / "imagesTr" / f"{case}_{i:04d}.nii.gz"
            if not link.is_symlink() and not link.exists():
                link.symlink_to(src / f"{case}-{m}.nii.gz")
        lab = out / "labelsTr" / f"{case}.nii.gz"
        if not lab.is_symlink() and not lab.exists():
            lab.symlink_to(src / f"{case}-seg.nii.gz")

    # Plain multi-class labels: no `regions_class_order`, so nnU-Net uses Dice+CE with a
    # softmax head — the same objective the replica builds.
    json.dump({
        "channel_names": {str(i): m.upper() for i, m in enumerate(MODALITIES)},
        "labels": {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4},
        "numTraining": len(train) + len(val),
        "file_ending": ".nii.gz",
    }, open(out / "dataset.json", "w"), indent=1)

    json.dump([{"train": train, "val": val}], open(SP / "cmp_splits_fold0.json", "w"), indent=1)

    print(f"Dataset500_ReplicaCmp -> {out}")
    print(f"  train {len(train)} cases / {len(set(map(patient, train)))} patients")
    print(f"  val   {len(val)} cases / {len(set(map(patient, val)))} patients")
    print(f"  channel order {MODALITIES}")


if __name__ == "__main__":
    sys.exit(main())

