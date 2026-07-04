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
from typing import Dict, List, Tuple, Union

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

CASE_PATTERN = re.compile(r"BraTS-GLI-(\d{5})-(\d{3})")

# A data source is either a single directory or several to pool together.
DataDirs = Union[str, Path, List[Union[str, Path]]]


def extract_patient_id(case_name: str) -> str:
    """Extract patient ID from BraTS case name."""
    match = CASE_PATTERN.match(case_name)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot parse patient ID from: {case_name}")


def resolve_train_dirs(data_cfg: Dict) -> List[str]:
    """Build the list of directories to draw training cases from.

    Combines the primary ``train_dir`` with any ``extra_train_dirs`` (e.g. an
    additional BraTS 2024 GLI training release) into one pool that is then split
    at patient level. ``train_dir`` may itself be a string or a list. Paths are
    expanded and de-duplicated while preserving order.

    Every entrypoint that reconstructs splits must call this so the (deterministic,
    seed-based) train/val/test partition is identical across training and evaluation.
    """
    train_dir = data_cfg.get("train_dir")
    dirs: List[Union[str, Path]] = list(train_dir) if isinstance(train_dir, (list, tuple)) else [train_dir]

    extra = data_cfg.get("extra_train_dirs") or []
    if isinstance(extra, (str, Path)):
        extra = [extra]
    dirs.extend(extra)

    seen, out = set(), []
    for d in dirs:
        if not d:
            continue
        p = str(Path(d).expanduser())
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _collect_case_dirs(data_dir: DataDirs) -> List[Path]:
    """Glob BraTS case directories from one source dir or several pooled together.

    Pooling lets an additional training set be merged before patient-level
    splitting. Case directories are matched by ``CASE_PATTERN`` and returned
    sorted by name (deterministic). If the same case name appears in more than
    one source dir, the first occurrence is kept and a warning is printed.
    """
    if isinstance(data_dir, (str, Path)):
        source_dirs = [data_dir]
    else:
        source_dirs = list(data_dir)

    by_name: Dict[str, Path] = {}
    duplicates: List[str] = []
    for d in source_dirs:
        dp = Path(d).expanduser()
        if not dp.is_dir():
            raise FileNotFoundError(f"Data directory not found: {dp}")
        for x in dp.iterdir():
            if not x.is_dir() or x.name.startswith(".") or not CASE_PATTERN.match(x.name):
                continue
            if x.name in by_name:
                duplicates.append(x.name)
                continue
            by_name[x.name] = x

    if duplicates:
        console.print(
            f"[yellow]Warning: {len(duplicates)} case name(s) appear in multiple "
            f"source dirs; kept first occurrence (e.g. {duplicates[0]}).[/yellow]"
        )

    return sorted(by_name.values(), key=lambda p: p.name)


def group_by_patient(case_dirs: List[Path]) -> Dict[str, List[Path]]:
    """Group case directories by patient ID."""
    patient_cases = defaultdict(list)
    for case_dir in case_dirs:
        pid = extract_patient_id(case_dir.name)
        patient_cases[pid].append(case_dir)
    return dict(patient_cases)


def create_patient_splits(
    data_dir: DataDirs,
    split_ratios: List[float] = [0.75, 0.15, 0.10],
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split dataset at patient level to prevent longitudinal leakage.

    ``data_dir`` may be a single directory or a list of directories to pool
    (e.g. main + additional BraTS training releases) before splitting.

    Returns:
        (train_cases, val_cases, test_cases) - lists of case directory paths
    """
    case_dirs = _collect_case_dirs(data_dir)

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


def create_kfold_splits(
    data_dir: DataDirs,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[Path], List[Path]]]:
    """Create K-fold cross-validation splits at patient level.

    Uses ALL data for training/validation (no held-out test set).
    Each fold's validation set contains ~1/K of patients. ``data_dir`` may be a
    single directory or a list of directories to pool before splitting.

    Returns:
        List of (train_cases, val_cases) tuples, one per fold.
    """
    case_dirs = _collect_case_dirs(data_dir)

    patient_cases = group_by_patient(case_dirs)
    patient_ids = sorted(patient_cases.keys())

    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)

    # Split patients into K folds
    fold_size = len(patient_ids) // n_folds
    folds = []

    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        if fold_idx == n_folds - 1:
            val_pids = patient_ids[start:]  # Last fold gets remainder
        else:
            val_pids = patient_ids[start:start + fold_size]
        train_pids = [p for p in patient_ids if p not in set(val_pids)]

        train_cases = sorted([c for pid in train_pids for c in patient_cases[pid]])
        val_cases = sorted([c for pid in val_pids for c in patient_cases[pid]])

        # Verify no overlap
        train_pid_set = set(extract_patient_id(c.name) for c in train_cases)
        val_pid_set = set(extract_patient_id(c.name) for c in val_cases)
        assert train_pid_set.isdisjoint(val_pid_set), f"Fold {fold_idx}: patient overlap!"

        folds.append((train_cases, val_cases))

    # Print summary
    table = Table(title=f"{n_folds}-Fold Cross-Validation (Patient-Level)", style="bold cyan")
    table.add_column("Fold", style="bold")
    table.add_column("Train Patients", justify="right")
    table.add_column("Train Cases", justify="right")
    table.add_column("Val Patients", justify="right")
    table.add_column("Val Cases", justify="right")
    for i, (tc, vc) in enumerate(folds):
        tp = len(set(extract_patient_id(c.name) for c in tc))
        vp = len(set(extract_patient_id(c.name) for c in vc))
        table.add_row(f"Fold {i}", str(tp), str(len(tc)), str(vp), str(len(vc)))
    table.add_row("Total", str(len(patient_ids)), str(len(case_dirs)), "", "")
    console.print(table)

    return folds


def create_kfold_splits_with_test(
    data_dir: DataDirs,
    n_folds: int = 5,
    seed: int = 42,
    split_ratios: List[float] = [0.75, 0.15, 0.10],
) -> Tuple[List[Tuple[List[Path], List[Path]]], List[Path]]:
    """Patient-level K-fold CV over the train+val pool, with a HELD-OUT test set.

    Unlike `create_kfold_splits` (which uses ALL data), this reserves a test set
    that NEVER appears in any fold's train or val. The reserved test patients are
    exactly `create_patient_splits(seed, split_ratios)`'s test split, so the test
    set lines up with the holdout pipeline and stays comparable across experiments.

    The remaining (train+val) patients are partitioned into `n_folds` folds.

    Returns:
        (folds, test_cases) where folds = list of (train_cases, val_cases).
    """
    case_dirs = _collect_case_dirs(data_dir)

    patient_cases = group_by_patient(case_dirs)
    patient_ids = sorted(patient_cases.keys())

    # Same shuffle/partition as create_patient_splits -> identical test patients.
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    pool_pids = patient_ids[:n_train + n_val]   # train+val patients -> get K-folded
    test_pids = patient_ids[n_train + n_val:]   # held out of every fold
    test_pid_set = set(test_pids)

    test_cases = sorted(
        [c for pid in test_pids for c in patient_cases[pid]], key=lambda p: p.name
    )

    # K-fold the pool (already shuffled) by chunking, same scheme as create_kfold_splits.
    fold_size = len(pool_pids) // n_folds
    folds: List[Tuple[List[Path], List[Path]]] = []
    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        if fold_idx == n_folds - 1:
            val_pids = pool_pids[start:]
        else:
            val_pids = pool_pids[start:start + fold_size]
        val_pid_set = set(val_pids)
        train_pids = [p for p in pool_pids if p not in val_pid_set]

        train_cases = sorted([c for pid in train_pids for c in patient_cases[pid]], key=lambda p: p.name)
        val_cases = sorted([c for pid in val_pids for c in patient_cases[pid]], key=lambda p: p.name)

        # No patient appears in both train and val, and NO test patient leaks in.
        assert val_pid_set.isdisjoint(set(train_pids)), f"Fold {fold_idx}: train/val overlap!"
        assert test_pid_set.isdisjoint(val_pid_set), f"Fold {fold_idx}: test leaked into val!"
        assert test_pid_set.isdisjoint(set(train_pids)), f"Fold {fold_idx}: test leaked into train!"
        folds.append((train_cases, val_cases))

    # Summary
    table = Table(title=f"{n_folds}-Fold CV + Held-out Test (Patient-Level)", style="bold cyan")
    for col in ("Fold", "Train Patients", "Train Cases", "Val Patients", "Val Cases"):
        table.add_column(col, style="bold" if col == "Fold" else None,
                         justify="left" if col == "Fold" else "right")
    for i, (tc, vc) in enumerate(folds):
        tp = len(set(extract_patient_id(c.name) for c in tc))
        vp = len(set(extract_patient_id(c.name) for c in vc))
        table.add_row(f"Fold {i}", str(tp), str(len(tc)), str(vp), str(len(vc)))
    table.add_row("[dim]Test (held out)[/dim]", str(len(test_pids)), str(len(test_cases)), "", "")
    table.add_row("Total", str(n), str(len(case_dirs)), "", "")
    console.print(table)

    return folds, test_cases
