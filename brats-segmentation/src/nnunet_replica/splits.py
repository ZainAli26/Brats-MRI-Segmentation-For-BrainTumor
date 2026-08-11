"""Turn the repo's patient-level splits into case-id lists for the replica.

The replica trains on a preprocessed *case cache* keyed by case id, so this is the one
place that converts ``src.data.splits``' directory-based folds into ids. Two things are
kept deliberately identical to every other experiment:

* the partition comes from ``split_seed`` (42) at **patient** level, so no BraTS-GLI
  patient ever straddles train and val;
* with ``kfold_holdout_test: true`` the seed-42 10 % test patients are reserved out of
  every fold — matching the native exp19 run, so the replica's held-out test numbers are
  directly comparable to it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.data.splits import (
    create_kfold_splits,
    create_kfold_splits_with_test,
    create_patient_splits,
    resolve_train_dirs,
)


def _ids(case_dirs: Sequence[Path]) -> List[str]:
    return [p.name for p in case_dirs]


def build_replica_splits(config: Dict) -> Tuple[List[Tuple[List[str], List[str]]], List[str], List[Path]]:
    """Return ``(folds_as_case_ids, test_case_ids, all_case_dirs)``.

    ``all_case_dirs`` is what the preprocessor needs (paths to the raw NIfTI dirs);
    the id lists are what the trainer needs.
    """
    data_cfg = config["data"]
    sources = resolve_train_dirs(data_cfg)
    n_folds = int(data_cfg.get("n_folds", 5))
    seed = int(data_cfg.get("split_seed", 42))
    holdout = bool(data_cfg.get("kfold_holdout_test", True))

    if holdout:
        folds, test_cases = create_kfold_splits_with_test(
            sources, n_folds=n_folds, seed=seed,
            split_ratios=data_cfg.get("split_ratios", [0.75, 0.15, 0.10]),
        )
    else:
        folds = create_kfold_splits(sources, n_folds=n_folds, seed=seed)
        test_cases = []

    fold_ids = [(_ids(tr), _ids(va)) for tr, va in folds]

    all_dirs = sorted(
        {c for tr, va in folds for c in list(tr) + list(va)} | set(test_cases),
        key=lambda p: p.name,
    )
    return fold_ids, _ids(test_cases), all_dirs


def save_splits(folds: Sequence[Tuple[List[str], List[str]]], test_ids: Sequence[str],
                path: str | Path) -> None:
    payload = [{"train": list(tr), "val": list(va)} for tr, va in folds]
    with open(path, "w") as f:
        json.dump({"folds": payload, "test": list(test_ids)}, f, indent=2)


def holdout_split_ids(config: Dict) -> Tuple[List[str], List[str], List[str]]:
    """The non-k-fold 75/15/10 partition, as case ids (used by smoke/overfit runs)."""
    data_cfg = config["data"]
    train, val, test = create_patient_splits(
        resolve_train_dirs(data_cfg),
        split_ratios=data_cfg.get("split_ratios", [0.75, 0.15, 0.10]),
        seed=int(data_cfg.get("split_seed", 42)),
    )
    return _ids(train), _ids(val), _ids(test)
