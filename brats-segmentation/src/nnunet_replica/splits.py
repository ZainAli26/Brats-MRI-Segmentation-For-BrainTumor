"""Turn the repo's patient-level splits into case-id lists for the replica.

The replica trains on a preprocessed *case cache* keyed by case id, so this is the one
place that converts ``src.data.splits``' directory-based folds into ids. Two things are
kept deliberately identical to every other experiment:

* the partition comes from ``split_seed`` (42) at **patient** level, so no BraTS-GLI
  patient ever straddles train and val;
* with ``kfold_holdout_test: true`` the seed-42 10 % test patients are reserved out of
  every fold — matching the native exp19 run, so the replica's held-out test numbers are
  directly comparable to it.

``data.synthetic_train_dirs`` adds registration-augmented cases (augment_registration.py)
to the **train side only** — see ``attach_synthetic_train_cases`` for the rules.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from rich.console import Console
from rich.table import Table

from src.data.splits import (
    CASE_PATTERN,
    create_kfold_splits,
    create_kfold_splits_with_test,
    create_patient_splits,
    extract_patient_id,
    resolve_train_dirs,
)

console = Console()

# A synthetic case written by augment_registration.py:
#   BraTS-GLI-<moving patient>-<moving tp>-reg-<fixed patient>-<fixed tp>
SYNTHETIC_CASE_PATTERN = re.compile(r"^BraTS-GLI-(\d{5})-(\d{3})-reg-(\d{5})-(\d{3})$")


class SyntheticCase(NamedTuple):
    """A registration-augmented case and the two real patients it is made of."""
    case_id: str
    path: Path
    moving_patient: str
    fixed_patient: str


def parse_synthetic_case(case_dir: Path) -> Optional[SyntheticCase]:
    """Recover both source patients from a synthetic case directory name, or None."""
    m = SYNTHETIC_CASE_PATTERN.match(case_dir.name)
    if not m:
        return None
    return SyntheticCase(case_dir.name, case_dir, m.group(1), m.group(3))


def _ids(case_dirs: Sequence[Path]) -> List[str]:
    return [p.name for p in case_dirs]


def collect_synthetic_case_dirs(dirs: Sequence[str | Path]) -> List[SyntheticCase]:
    """Every parseable synthetic case directory under the given roots, sorted by id."""
    found: Dict[str, SyntheticCase] = {}
    unparseable: List[str] = []
    for d in dirs:
        root = Path(d).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"synthetic_train_dirs entry not found: {root}")
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            case = parse_synthetic_case(child)
            if case is None:
                unparseable.append(child.name)
            elif case.case_id not in found:
                found[case.case_id] = case
    if unparseable:
        console.print(
            f"[yellow]{len(unparseable)} directory/ies in synthetic_train_dirs do not "
            f"match the '<case>-reg-<case>' naming and were ignored "
            f"(e.g. {unparseable[0]}).[/yellow]"
        )
    return [found[k] for k in sorted(found)]


def assert_no_synthetic_in_real_pool(case_dirs: Sequence[Path]) -> None:
    """Refuse to run if synthetic cases were pooled in as if they were real data.

    Putting the synthetic dir in ``train_dir``/``extra_train_dirs`` would patient-split
    it like real data and drop synthetic cases into validation and test sets, quietly
    invalidating every number the run produces. That mistake dies here.
    """
    bad = sorted(p.name for p in case_dirs if SYNTHETIC_CASE_PATTERN.match(p.name))
    if bad:
        raise ValueError(
            f"{len(bad)} synthetic case(s) are in the REAL training pool (e.g. {bad[0]}). "
            "Registration-augmented data must be listed under data.synthetic_train_dirs, "
            "never train_dir / extra_train_dirs — otherwise it lands in val/test."
        )


def attach_synthetic_train_cases(
    folds: Sequence[Tuple[List[Path], List[Path]]],
    synthetic: Sequence[SyntheticCase],
    verbose: bool = True,
) -> List[Tuple[List[Path], List[Path]]]:
    """Add synthetic cases to each fold's TRAIN list, never to val.

    A synthetic case carries the *moving* patient's tumour and labels warped onto the
    *fixed* patient's anatomy, so it is only admissible for a fold whose train set owns
    BOTH patients. If either one is that fold's validation patient — or a held-out test
    patient, who is in no fold's train set — the case is dropped for that fold. So the
    same synthetic pool is filtered differently per fold, and every fold's val set stays
    exactly the real-data val set the other experiments use.
    """
    out: List[Tuple[List[Path], List[Path]]] = []
    rows: List[Tuple[int, int, int]] = []
    for fold_idx, (train_cases, val_cases) in enumerate(folds):
        train_pids = {extract_patient_id(p.name) for p in train_cases}
        eligible = [
            s for s in synthetic
            if s.moving_patient in train_pids and s.fixed_patient in train_pids
        ]
        # Sorted so the trainer sees a deterministic case order.
        augmented = sorted(list(train_cases) + [s.path for s in eligible],
                           key=lambda p: p.name)
        out.append((augmented, list(val_cases)))
        rows.append((fold_idx, len(train_cases), len(eligible)))

    if verbose:
        table = Table(title="Registration-augmented training cases", style="bold cyan")
        table.add_column("Fold", style="bold")
        table.add_column("Real train", justify="right")
        table.add_column("+ Synthetic", justify="right")
        table.add_column("Total train", justify="right")
        table.add_column("Dropped (patient not in fold train)", justify="right")
        for fold_idx, n_real, n_synth in rows:
            table.add_row(f"Fold {fold_idx}", str(n_real), str(n_synth),
                          str(n_real + n_synth), str(len(synthetic) - n_synth))
        table.add_row("[dim]Val / test[/dim]", "[dim]real only[/dim]", "[dim]0[/dim]",
                      "", "")
        console.print(table)
    return out


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

    # The real partition is fixed at this point — identical to every other experiment.
    real_cases = {c for tr, va in folds for c in list(tr) + list(va)} | set(test_cases)
    assert_no_synthetic_in_real_pool(sorted(real_cases, key=lambda p: p.name))

    synthetic_dirs = data_cfg.get("synthetic_train_dirs") or []
    if isinstance(synthetic_dirs, (str, Path)):
        synthetic_dirs = [synthetic_dirs]
    if synthetic_dirs:
        overlap = {str(Path(d).expanduser()) for d in synthetic_dirs} & set(sources)
        if overlap:
            raise ValueError(
                f"synthetic_train_dirs and the real training dirs overlap: {sorted(overlap)}. "
                "Synthetic data must live in its own directory."
            )
        folds = attach_synthetic_train_cases(folds, collect_synthetic_case_dirs(synthetic_dirs))

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
