"""Patient-level dataset splitting for BraTS longitudinal data.

BraTS case IDs follow the pattern: BraTS-GLI-XXXXX-YYY
  XXXXX = patient ID (shared across timepoints)
  YYY   = timepoint index

This module ensures all scans from the same patient stay in the same split,
preventing data leakage from longitudinal cases.
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

CASE_PATTERN = re.compile(r"BraTS-GLI-(\d{5})-(\d{3})")


def extract_patient_id(case_name: str) -> str:
    """Extract patient ID from BraTS case name."""
    match = CASE_PATTERN.match(case_name)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot parse patient ID from: {case_name}")


def group_by_patient(case_dirs: List[Path]) -> Dict[str, List[Path]]:
    """Group case directories by patient ID."""
    patient_cases = defaultdict(list)
    for case_dir in case_dirs:
        pid = extract_patient_id(case_dir.name)
        patient_cases[pid].append(case_dir)
    return dict(patient_cases)


def create_patient_splits(
    data_dir: str,
    split_ratios: List[float] = [0.75, 0.15, 0.10],
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split dataset at patient level to prevent longitudinal leakage.

    Returns:
        (train_cases, val_cases, test_cases) - lists of case directory paths
    """
    data_path = Path(data_dir).expanduser()
    case_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".") and CASE_PATTERN.match(d.name)
    ])

    patient_cases = group_by_patient(case_dirs)
    patient_ids = sorted(patient_cases.keys())

    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    train_pids = patient_ids[:n_train]
    val_pids = patient_ids[n_train:n_train + n_val]
    test_pids = patient_ids[n_train + n_val:]

    train_cases = [c for pid in train_pids for c in patient_cases[pid]]
    val_cases = [c for pid in val_pids for c in patient_cases[pid]]
    test_cases = [c for pid in test_pids for c in patient_cases[pid]]

    # Print split summary
    table = Table(title="Patient-Level Data Split", style="bold cyan")
    table.add_column("Split", style="bold")
    table.add_column("Patients", justify="right")
    table.add_column("Cases", justify="right")
    table.add_row("Train", str(len(train_pids)), str(len(train_cases)))
    table.add_row("Val", str(len(val_pids)), str(len(val_cases)))
    table.add_row("Test", str(len(test_pids)), str(len(test_cases)))
    table.add_row("Total", str(n), str(len(case_dirs)))
    console.print(table)

    # Verify no patient overlap
    assert set(train_pids).isdisjoint(set(val_pids)), "Train/val patient overlap!"
    assert set(train_pids).isdisjoint(set(test_pids)), "Train/test patient overlap!"
    assert set(val_pids).isdisjoint(set(test_pids)), "Val/test patient overlap!"

    return sorted(train_cases), sorted(val_cases), sorted(test_cases)
