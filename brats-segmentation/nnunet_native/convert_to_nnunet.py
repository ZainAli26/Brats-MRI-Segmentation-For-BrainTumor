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
  2. Remaps labels (4 -> 3) so classes are contiguous
  3. Generates dataset.json
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
from src.data.splits import create_patient_splits, extract_patient_id

console = Console()

# BraTS 2024 modality suffixes -> nnU-Net channel index
MODALITY_MAP = {
    "t1n": "0000",  # T1 native
    "t1c": "0001",  # T1 contrast-enhanced
    "t2w": "0002",  # T2 weighted
    "t2f": "0003",  # T2 FLAIR
}

# Remap label 4 -> 3 for contiguous classes
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}


def remap_and_save_label(src_path: Path, dst_path: Path):
    """Load segmentation, remap labels to contiguous, and save."""
    img = nib.load(str(src_path))
    data = img.get_fdata().astype(np.uint8)
    remapped = np.zeros_like(data)
    for src_label, dst_label in LABEL_REMAP.items():
        remapped[data == src_label] = dst_label
    out = nib.Nifti1Image(remapped, img.affine, img.header)
    nib.save(out, str(dst_path))


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
        "labels": {
            "background": 0,
            "NCR": 1,
            "ED": 2,
            "ET": 3,
        },
        "numTraining": len(training_entries),
        "file_ending": ".nii.gz",
        "regions_class_order": [1, 2, 3],
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
    args = parser.parse_args()

    convert_dataset(args.data_dir, args.output_dir, args.dataset_id, args.dataset_name)
