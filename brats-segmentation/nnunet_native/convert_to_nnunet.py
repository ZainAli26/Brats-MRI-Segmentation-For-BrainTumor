#!/usr/bin/env python3
"""Convert BraTS 2024 data to nnU-Net v2 raw format.

nnU-Net v2 expects:
    nnUNet_raw/
      DatasetXXX_Name/
        imagesTr/   {case}_0000.nii.gz, {case}_0001.nii.gz, ...
        labelsTr/   {case}.nii.gz
        imagesTs/   (optional)
        dataset.json

This script:
  1. Symlinks BraTS NIfTI files into nnU-Net's expected layout
  2. Remaps labels (identity for BraTS 2024 — already contiguous 0..4)
  3. Generates dataset.json (5-class post-treatment scheme)
  4. Creates a custom split file (splits_final.json) that respects
     patient-level grouping to prevent longitudinal leakage

Usage:
    python nnunet_native/convert_to_nnunet.py \
        --data_dir ../Brats2024/training_data1_v2 \
        --output_dir ./nnunet_data \
        --dataset_id 101 \
        --dataset_name BraTS2024
"""

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm
from rich.console import Console

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.splits import (
    create_patient_splits,
    create_kfold_splits_with_test,
    extract_patient_id,
)

console = Console()

# BraTS 2024 modality suffixes -> nnU-Net channel index
MODALITY_MAP = {
    "t1n": "0000",  # T1 native
    "t1c": "0001",  # T1 contrast-enhanced
    "t2w": "0002",  # T2 weighted
    "t2f": "0003",  # T2 FLAIR
}

# BraTS 2024 post-treatment labels are already contiguous (0=bg, 1=NETC, 2=SNFH,
# 3=ET, 4=RC), so the remap is the identity. (The old {0:0,1:1,2:2,4:3} map was for
# the pre-2024 scheme and silently DELETED label 3 / collided label 4 on this data.)
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

# Canonical 5-class label table shared by dataset.json below.
DATASET_LABELS = {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4}
REGIONS_CLASS_ORDER = [1, 2, 3, 4]


def remap_and_save_label(src_path: Path, dst_path: Path):
    """Load segmentation, remap labels to contiguous, and save."""
    img = nib.load(str(src_path))
    data = img.get_fdata().astype(np.uint8)
    remapped = np.zeros_like(data)
    for src_label, dst_label in LABEL_REMAP.items():
        remapped[data == src_label] = dst_label
    out = nib.Nifti1Image(remapped, img.affine, img.header)
    nib.save(out, str(dst_path))


def _link_case_images(case_dir: Path, images_dir: Path) -> bool:
    """Symlink all 4 modalities for a case into nnU-Net layout.

    Returns True if all modalities were found and linked, False otherwise.
    """
    for mod_suffix, channel_idx in MODALITY_MAP.items():
        src = list(case_dir.glob(f"*-{mod_suffix}.nii.gz"))
        if not src:
            console.print(f"[yellow]Missing {mod_suffix} for {case_dir.name}, skipping[/yellow]")
            return False
        dst = images_dir / f"{case_dir.name}_{channel_idx}.nii.gz"
        if not dst.exists():
            os.symlink(str(src[0].resolve()), str(dst))
    return True


def convert_dataset_kfold(
    data_dir: str,
    output_dir: str,
    dataset_id: int = 102,
    dataset_name: str = "BraTS2024ResEnc",
    n_folds: int = 5,
    split_seed: int = 42,
    split_ratios: list = [0.75, 0.15, 0.10],
):
    """Convert BraTS data to nnU-Net format for K-fold CV with a HELD-OUT test set.

    The test patients (same seed/ratios as create_patient_splits) are reserved in
    imagesTs and NEVER appear in any fold's train or val. The remaining train+val
    patients go into imagesTr and are partitioned into `n_folds` patient-level folds
    via splits_final.json. The 5-fold out-of-fold validation predictions give the CV
    metric; the held-out test set is for final (leak-free) evaluation.
    """
    data_path = Path(data_dir).expanduser().resolve()
    base_dir = Path(output_dir).expanduser().resolve()

    raw_dir = base_dir / "nnUNet_raw" / f"Dataset{dataset_id:03d}_{dataset_name}"
    preprocessed_dir = base_dir / "nnUNet_preprocessed"
    results_dir = base_dir / "nnUNet_results"
    images_tr = raw_dir / "imagesTr"
    labels_tr = raw_dir / "labelsTr"
    images_ts = raw_dir / "imagesTs"

    for d in [images_tr, labels_tr, images_ts, preprocessed_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # K-fold over the train+val pool, with the test set held out of every fold.
    folds, test_cases = create_kfold_splits_with_test(
        str(data_path), n_folds=n_folds, seed=split_seed, split_ratios=split_ratios
    )
    test_case_set = {c.name for c in test_cases}

    # Pool cases (train+val across folds) == everything that is NOT test.
    pool_cases = sorted({c for tc, vc in folds for c in (tc + vc)}, key=lambda p: p.name)

    console.print(f"\n[bold]Converting {len(pool_cases)} train+val cases to imagesTr...[/bold]")
    training_entries = []
    for case_dir in tqdm(pool_cases, desc="Train+Val"):
        case_id = case_dir.name
        assert case_id not in test_case_set, f"{case_id} is a test case — must not be in imagesTr!"
        if not _link_case_images(case_dir, images_tr):
            continue
        seg_src = list(case_dir.glob("*-seg.nii.gz"))
        if not seg_src:
            console.print(f"[yellow]No seg for {case_id}, skipping[/yellow]")
            continue
        dst = labels_tr / f"{case_id}.nii.gz"
        if not dst.exists():
            remap_and_save_label(seg_src[0], dst)
        training_entries.append(case_id)

    # Held-out test set -> imagesTs only (no labels here; GT is read from data_dir at eval).
    console.print(f"[bold]Linking {len(test_cases)} held-out test cases to imagesTs...[/bold]")
    for case_dir in tqdm(test_cases, desc="Test"):
        _link_case_images(case_dir, images_ts)

    # --- dataset.json ---
    dataset_json = {
        "channel_names": {"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2f"},
        "labels": dict(DATASET_LABELS),
        "numTraining": len(training_entries),
        "file_ending": ".nii.gz",
        "regions_class_order": list(REGIONS_CLASS_ORDER),
    }
    with open(raw_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # --- splits_final.json: one entry per fold (case-id level) ---
    # Built only from imagesTr cases -> test cases can never enter a fold.
    entry_set = set(training_entries)
    splits = []
    for fold_idx, (tc, vc) in enumerate(folds):
        train_ids = sorted(c.name for c in tc if c.name in entry_set)
        val_ids = sorted(c.name for c in vc if c.name in entry_set)
        train_pids = set(extract_patient_id(c) for c in train_ids)
        val_pids = set(extract_patient_id(c) for c in val_ids)
        test_pids = set(extract_patient_id(c) for c in test_case_set)
        assert train_pids.isdisjoint(val_pids), f"Fold {fold_idx}: train/val patient overlap!"
        assert test_pids.isdisjoint(train_pids), f"Fold {fold_idx}: test patient in train!"
        assert test_pids.isdisjoint(val_pids), f"Fold {fold_idx}: test patient in val!"
        splits.append({"train": train_ids, "val": val_ids})

    for target_dir in [raw_dir, preprocessed_dir / f"Dataset{dataset_id:03d}_{dataset_name}"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / "splits_final.json", "w") as f:
            json.dump(splits, f, indent=2)

    console.print(f"\n[bold green]K-fold conversion complete![/bold green]")
    console.print(f"  Raw data:     {raw_dir}")
    console.print(f"  Train+val:    {len(training_entries)} cases ({n_folds}-fold, patient-level, seed={split_seed})")
    console.print(f"  Test (held out, never trained): {len(test_cases)} cases -> imagesTs")
    console.print(f"\n[bold]Export before running nnU-Net:[/bold]")
    console.print(f"  export nnUNet_raw={base_dir / 'nnUNet_raw'}")
    console.print(f"  export nnUNet_preprocessed={preprocessed_dir}")
    console.print(f"  export nnUNet_results={results_dir}")

    return raw_dir, preprocessed_dir, results_dir


def convert_dataset(
    data_dir: str,
    output_dir: str,
    dataset_id: int = 101,
    dataset_name: str = "BraTS2024",
    split_ratios: list = [0.75, 0.15, 0.10],
    split_seed: int = 42,
):
    data_path = Path(data_dir).expanduser().resolve()
    base_dir = Path(output_dir).expanduser().resolve()

    raw_dir = base_dir / "nnUNet_raw" / f"Dataset{dataset_id:03d}_{dataset_name}"
    preprocessed_dir = base_dir / "nnUNet_preprocessed"
    results_dir = base_dir / "nnUNet_results"

    images_tr = raw_dir / "imagesTr"
    labels_tr = raw_dir / "labelsTr"
    images_ts = raw_dir / "imagesTs"

    for d in [images_tr, labels_tr, images_ts, preprocessed_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get patient-level splits (same seed as our custom pipeline)
    train_cases, val_cases, test_cases = create_patient_splits(
        str(data_path), split_ratios, split_seed
    )

    # nnU-Net uses all labeled data for training with its own cross-validation.
    # We put train+val into imagesTr/labelsTr and test into imagesTs.
    # Then we provide a custom splits_final.json that matches our patient-level split.
    trainval_cases = sorted(train_cases + val_cases)
    test_case_set = set(c.name for c in test_cases)

    console.print(f"\n[bold]Converting {len(trainval_cases)} train+val cases to nnU-Net format...[/bold]")

    training_entries = []
    for case_dir in tqdm(trainval_cases, desc="Train+Val"):
        case_id = case_dir.name

        # Symlink modality files
        for mod_suffix, channel_idx in MODALITY_MAP.items():
            src = list(case_dir.glob(f"*-{mod_suffix}.nii.gz"))
            if not src:
                console.print(f"[yellow]Missing {mod_suffix} for {case_id}, skipping[/yellow]")
                break
            dst = images_tr / f"{case_id}_{channel_idx}.nii.gz"
            if not dst.exists():
                os.symlink(str(src[0].resolve()), str(dst))
        else:
            # Remap and save label
            seg_src = list(case_dir.glob("*-seg.nii.gz"))
            if seg_src:
                dst = labels_tr / f"{case_id}.nii.gz"
                if not dst.exists():
                    remap_and_save_label(seg_src[0], dst)
                training_entries.append({"image": f"./imagesTr/{case_id}.nii.gz",
                                         "label": f"./labelsTr/{case_id}.nii.gz"})

    console.print(f"[bold]Converting {len(test_cases)} test cases...[/bold]")
    for case_dir in tqdm(test_cases, desc="Test"):
        case_id = case_dir.name
        for mod_suffix, channel_idx in MODALITY_MAP.items():
            src = list(case_dir.glob(f"*-{mod_suffix}.nii.gz"))
            if not src:
                break
            dst = images_ts / f"{case_id}_{channel_idx}.nii.gz"
            if not dst.exists():
                os.symlink(str(src[0].resolve()), str(dst))

    # --- dataset.json ---
    dataset_json = {
        "channel_names": {
            "0": "T1n",
            "1": "T1c",
            "2": "T2w",
            "3": "T2f",
        },
        "labels": dict(DATASET_LABELS),
        "numTraining": len(training_entries),
        "file_ending": ".nii.gz",
        "regions_class_order": list(REGIONS_CLASS_ORDER),
    }

    with open(raw_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # --- Custom splits_final.json for patient-level splitting ---
    # nnU-Net reads this from the preprocessed folder to override its random splits.
    # We create a single fold where train=train_cases, val=val_cases.
    train_ids = [c.name for c in train_cases if c.name not in test_case_set]
    val_ids = [c.name for c in val_cases if c.name not in test_case_set]

    # Verify no patient overlap
    train_pids = set(extract_patient_id(c) for c in train_ids)
    val_pids = set(extract_patient_id(c) for c in val_ids)
    assert train_pids.isdisjoint(val_pids), "Patient overlap between train and val!"

    splits = [{"train": sorted(train_ids), "val": sorted(val_ids)}]

    # Save splits to both raw and preprocessed (nnU-Net checks preprocessed first)
    for target_dir in [raw_dir, preprocessed_dir / f"Dataset{dataset_id:03d}_{dataset_name}"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / "splits_final.json", "w") as f:
            json.dump(splits, f, indent=2)

    console.print(f"\n[bold green]Conversion complete![/bold green]")
    console.print(f"  Raw data:       {raw_dir}")
    console.print(f"  Preprocessed:   {preprocessed_dir}")
    console.print(f"  Results:        {results_dir}")
    console.print(f"  Train cases:    {len(train_ids)}")
    console.print(f"  Val cases:      {len(val_ids)}")
    console.print(f"  Test cases:     {len(test_cases)}")
    console.print(f"\n[bold]Set these environment variables before running nnU-Net:[/bold]")
    console.print(f"  export nnUNet_raw={base_dir / 'nnUNet_raw'}")
    console.print(f"  export nnUNet_preprocessed={preprocessed_dir}")
    console.print(f"  export nnUNet_results={results_dir}")

    return raw_dir, preprocessed_dir, results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BraTS data to nnU-Net v2 format")
    parser.add_argument("--data_dir", default="../Brats2024/training_data1_v2", help="BraTS data directory")
    parser.add_argument("--output_dir", default="./nnunet_data", help="nnU-Net output base directory")
    parser.add_argument("--dataset_id", type=int, default=101, help="nnU-Net dataset ID")
    parser.add_argument("--dataset_name", default="BraTS2024", help="nnU-Net dataset name")
    parser.add_argument("--mode", choices=["holdout", "kfold"], default="holdout",
                        help="holdout = 75/15/10 single split (Phase 5); "
                             "kfold = all data + N-fold patient split (exp19/exp18 replica)")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds (kfold mode only)")
    parser.add_argument("--split_seed", type=int, default=42, help="Patient-split seed")
    args = parser.parse_args()

    if args.mode == "kfold":
        convert_dataset_kfold(
            args.data_dir, args.output_dir, args.dataset_id, args.dataset_name,
            n_folds=args.n_folds, split_seed=args.split_seed,
        )
    else:
        convert_dataset(args.data_dir, args.output_dir, args.dataset_id, args.dataset_name)
