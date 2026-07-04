#!/usr/bin/env python3
"""Build a small nnU-Net dataset for a native-nnU-Net OVERFIT sanity check.

Copies N cases out of an already-converted nnU-Net dataset (default: the exp19
Dataset102) into a new tiny dataset. Only cases that actually contain ET
(label 3) are selected by default, so the overfit test exercises the hard
small classes (ET / tumor core), not just the big ones (SNFH / WT).

After running this, `val` is meant to equal `train` (write splits_final.json so
both lists are identical) and you train a short trainer — pseudo-Dice for ALL
classes should climb toward ~1.0 if the native pipeline is healthy.

Usage:
    python nnunet_native/make_overfit_dataset.py            # 50 ET cases -> Dataset199_Overfit
    python nnunet_native/make_overfit_dataset.py --num_cases 30
    python nnunet_native/make_overfit_dataset.py --no_require_et
"""

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from rich.console import Console

console = Console()

# BraTS 2024 post-treatment 5-class scheme (already contiguous).
DATASET_LABELS = {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4}
CHANNELS = ("0000", "0001", "0002", "0003")  # T1n, T1c, T2w, T2f
ET_LABEL = 3


def build_overfit_dataset(src_dir: str, dst_dir: str, num_cases: int, require_et: bool):
    src = Path(src_dir)
    dst = Path(dst_dir)
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)

    label_files = sorted(glob.glob(str(src / "labelsTr" / "*.nii.gz")))
    if not label_files:
        raise SystemExit(f"No labels found in {src/'labelsTr'} — is {src.name} converted?")

    picked = []
    for lab in label_files:
        case_id = os.path.basename(lab)[: -len(".nii.gz")]
        if require_et:
            vals = np.unique(nib.load(lab).get_fdata().astype("uint8"))
            if ET_LABEL not in vals:
                continue
        picked.append(case_id)
        if len(picked) == num_cases:
            break

    if len(picked) < num_cases:
        console.print(
            f"[yellow]Only found {len(picked)} matching cases "
            f"(requested {num_cases}{' with ET' if require_et else ''}).[/yellow]"
        )

    for case_id in picked:
        for ch in CHANNELS:
            s = src / "imagesTr" / f"{case_id}_{ch}.nii.gz"
            # imagesTr entries may be symlinks into the raw BraTS tree -> dereference.
            shutil.copyfile(os.path.realpath(s), dst / "imagesTr" / f"{case_id}_{ch}.nii.gz")
        shutil.copyfile(src / "labelsTr" / f"{case_id}.nii.gz",
                        dst / "labelsTr" / f"{case_id}.nii.gz")

    dataset_json = {
        "channel_names": {"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2f"},
        "labels": dict(DATASET_LABELS),
        "numTraining": len(picked),
        "file_ending": ".nii.gz",
    }
    with open(dst / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    console.print(f"[bold green]Built overfit dataset:[/bold green] {dst}")
    console.print(f"  Cases ({len(picked)}): {', '.join(picked)}")
    console.print("\n[bold]Next:[/bold]")
    console.print(f"  nnUNetv2_plan_and_preprocess -d {_dataset_id(dst)} -pl ResEncUNetPlanner "
                  f"-gpu_memory_target 7 -c 3d_fullres --verify_dataset_integrity")
    console.print("  # then write splits_final.json with train == val (see Step 3)")


def _dataset_id(dst: Path) -> str:
    name = dst.name  # e.g. Dataset199_Overfit
    digits = name.replace("Dataset", "")[:3]
    return digits if digits.isdigit() else "199"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a tiny nnU-Net overfit dataset")
    parser.add_argument("--src", default="nnunet_data/nnUNet_raw/Dataset102_BraTS2024ResEnc",
                        help="Source converted nnU-Net dataset (with imagesTr/labelsTr)")
    parser.add_argument("--dst", default="nnunet_data/nnUNet_raw/Dataset199_Overfit",
                        help="Destination tiny dataset dir (DatasetXXX_Name)")
    parser.add_argument("--num_cases", type=int, default=50, help="How many cases to copy")
    parser.add_argument("--no_require_et", dest="require_et", action="store_false",
                        help="Do not require label 3 (ET) to be present in selected cases")
    args = parser.parse_args()

    build_overfit_dataset(args.src, args.dst, args.num_cases, args.require_et)
