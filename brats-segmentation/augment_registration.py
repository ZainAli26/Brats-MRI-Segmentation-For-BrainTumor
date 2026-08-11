#!/usr/bin/env python3
"""Cross-patient registration augmentation for BraTS post-treatment glioma (GLI).

Implements the registration-based augmentation of Ferreira et al., "How we won BraTS
2023": take two DIFFERENT patients, deformably register the *moving* patient's brain
into the *fixed* patient's anatomical space, and apply that one transform to every
modality **and** to the segmentation. The result is a new, anatomically plausible
image+label training pair — the moving patient's tumour and labels, re-shaped onto
another brain's geometry.

    moving patient M ──SyN(t1n)──▶ fixed patient F's space  ⇒  BraTS-GLI-<M>-reg-<F>/

WHAT IS AND IS NOT SHARED
  The transform is estimated ONCE per pair, on the t1n modality only, and then reused
  verbatim for t1c/t2w/t2f and for the label. Estimating per modality would break the
  voxel correspondence between channels and label. Images are warped with
  ``interpolator='linear'``; the label is warped with ``interpolator='genericLabel'``
  (a multi-label interpolator) and is then checked to contain only the integer values
  that were in the source segmentation — a linear interpolation of a label map would
  invent fractional/intermediate classes and is never used here.

OUTPUT
  One directory per synthetic case, named ``BraTS-GLI-<moving>-reg-<fixed>`` where
  <moving> is the full moving case id and <fixed> is the fixed case's short
  ``NNNNN-TTT`` id, e.g.

      BraTS-GLI-00005-100-reg-00123-101/
          BraTS-GLI-00005-100-reg-00123-101-t1n.nii.gz
          ...-t1c.nii.gz  ...-t2w.nii.gz  ...-t2f.nii.gz  ...-seg.nii.gz

  The leading ``BraTS-GLI-<moving>`` keeps the repo's ``CASE_PATTERN`` (src/data/splits.py)
  matching, so the synthetic case is attributed to the MOVING patient — which is whose
  tumour and labels it actually carries. Both source patients are recoverable from the
  name (and from ``registration_manifest.json``), which is what lets the replica's
  splitter keep synthetic data out of any fold whose train set does not own BOTH
  patients (see ``src/nnunet_replica/splits.py``).

DATA LEAKAGE — READ THIS
  A synthetic case carries the moving patient's tumour and the fixed patient's brain
  geometry, so it must not be trained on if EITHER patient is being validated or tested.
  Two guards:
    1. here — pass ``--config <experiment.yaml>`` and the seed-42 held-out test patients
       are excluded from pairing entirely, in both roles;
    2. at train time — the replica splitter only adds a synthetic case to fold k's TRAIN
       list when both its patients are in fold k's train set, and never puts synthetic
       data in a val or test list.
  Keep the output in its OWN directory (``data.synthetic_train_dirs``), never in
  ``extra_train_dirs`` — the latter is patient-split like real data and would put
  synthetic cases into validation.

USAGE (run from brats-segmentation/)
    # quick smoke test with the fast transform
    python augment_registration.py --data-dir ../Brats2024/training_data1_v2 \
        --output-dir ../Brats2024/synthetic_regaug --transform Affine --max-pairs 2

    # the real pass: k=2 partners per case, SyN, test patients excluded
    python augment_registration.py \
        --data-dir ../Brats2024/training_data1_v2 \
        --data-dir ../Brats2024/training_data_additional \
        --output-dir ../Brats2024/synthetic_regaug \
        --config experiments/exp27_replica_5ch_regaug_5fold.yaml \
        -k 2 --seed 42 --num-workers 4 --with-subtraction

Re-running skips pairs that are already complete, so an interrupted pass resumes.

COST, measured on this dataset (182x218x182 @ 1 mm, 16-core box, --num-workers 4
--ants-threads 4): SyN takes ~180 s of CPU per pair, ~14 s of wall clock per pair at that
concurrency, and writes ~27 MB per case with --with-subtraction. The full k=2 pass over the
1621-case pool (2936 pairs) is therefore roughly 11 h and 79 GB of NIfTI, before the
preprocessed cache those cases will also need. Halve both with ``-k 1``; check free disk
before starting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import traceback
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import nibabel as nib
from rich.console import Console
from rich.progress import (
    BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table

# --- Dataset conventions (change here if a release uses different suffixes) ----------
MODALITY_SUFFIXES: Tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f")
REGISTRATION_MODALITY: str = "t1n"      # the one modality the transform is estimated on
LABEL_SUFFIX: str = "seg"
SUBTRACTION_SUFFIX: str = "sub"         # optional 5th channel, see --with-subtraction
FILE_EXT: str = ".nii.gz"
VALID_LABELS: frozenset = frozenset({0, 1, 2, 3, 4})   # NETC/SNFH/ET/RC + background
LABEL_DTYPE = np.uint8
SYNTHETIC_TAG: str = "reg"
MANIFEST_NAME: str = "registration_manifest.json"

# BraTS-GLI-XXXXX-YYY : XXXXX = patient, YYY = timepoint.
CASE_RE = re.compile(r"^BraTS-GLI-(\d{5})-(\d{3})$")
# A case this script produced. Groups: full moving id, moving patient, fixed short id,
# fixed patient. Must stay in step with SYNTHETIC_CASE_PATTERN in nnunet_replica/splits.py.
SYNTHETIC_CASE_RE = re.compile(r"^BraTS-GLI-((\d{5})-\d{3})-reg-((\d{5})-\d{3})$")

console = Console()


# =====================================================================================
# ANTsPy import — optional at module level so --help and the pairing logic work without it
# =====================================================================================

def _repair_stale_mpl_toolkits() -> bool:
    """Point ``mpl_toolkits`` at the matplotlib that is actually in use.

    Debian/Ubuntu's python3-matplotlib deb installs
    ``/usr/lib/python3/dist-packages/matplotlib-3.5.1-nspkg.pth``, which binds
    ``mpl_toolkits`` to the system copy at interpreter startup — before sys.path order
    can matter, so a newer pip matplotlib in ~/.local never gets a look in. Importing
    ``ants`` then runs matplotlib-3.5-era ``mpl_toolkits.axes_grid1`` against a 3.6+
    matplotlib and fails on the long-removed ``matplotlib.docstring``.

    Rebinding the name to the sibling of the matplotlib actually being imported fixes it
    for this process only. Returns True if anything was changed. The permanent fix is to
    remove that stale .pth — see the message in ``load_ants``.
    """
    # find_spec, not import: locating matplotlib must not execute it, or it warns about
    # the very breakage we are here to fix before we get the chance to fix it.
    try:
        spec = importlib.util.find_spec("matplotlib")
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.origin:
        return False
    expected = Path(spec.origin).resolve().parent.parent / "mpl_toolkits"
    if not expected.is_dir():
        return False
    current = list(getattr(sys.modules.get("mpl_toolkits"), "__path__", []))
    if current == [str(expected)]:
        return False
    package = types.ModuleType("mpl_toolkits")
    package.__path__ = [str(expected)]
    sys.modules["mpl_toolkits"] = package
    for name in [n for n in sys.modules if n.startswith("mpl_toolkits.")]:
        del sys.modules[name]
    return True


def load_ants():
    """Import ANTsPy, or exit with an actionable message."""
    _repair_stale_mpl_toolkits()
    try:
        import ants  # noqa: PLC0415  (deliberately lazy)
        return ants
    except ImportError as exc:
        first_error = exc

    missing = "No module named 'ants'" in str(first_error)
    if missing:
        console.print(
            "[red bold]ANTsPy is not installed.[/red bold]\n"
            f"  import error: {first_error}\n"
            "  install it with:  pip install 'antspyx>=0.4'\n"
            "  (it is in requirements.txt / the Docker image; this is the pip-installable\n"
            "   Python binding, not the command-line ANTs distribution)"
        )
    else:
        console.print(
            "[red bold]ANTsPy is installed but cannot be imported.[/red bold]\n"
            f"  import error: {first_error}\n"
            "  This is an environment problem, not a missing package. On Debian/Ubuntu it\n"
            "  is usually the system matplotlib shadowing a newer pip one:\n"
            "    sudo mv /usr/lib/python3/dist-packages/matplotlib-3.5.1-nspkg.pth \\\n"
            "            /usr/lib/python3/dist-packages/matplotlib-3.5.1-nspkg.pth.disabled\n"
            "  Running inside the Docker image avoids it entirely."
        )
    raise SystemExit(2)


# =====================================================================================
# Case discovery and pairing
# =====================================================================================

def patient_of(case_id: str) -> str:
    """'BraTS-GLI-00005-100' -> '00005'."""
    m = CASE_RE.match(case_id)
    if not m:
        raise ValueError(f"not a BraTS-GLI case id: {case_id}")
    return m.group(1)


def short_id(case_id: str) -> str:
    """'BraTS-GLI-00005-100' -> '00005-100'."""
    m = CASE_RE.match(case_id)
    if not m:
        raise ValueError(f"not a BraTS-GLI case id: {case_id}")
    return f"{m.group(1)}-{m.group(2)}"


def synthetic_case_name(moving_id: str, fixed_id: str) -> str:
    """The output directory / file prefix for a (moving, fixed) pair."""
    return f"{moving_id}-{SYNTHETIC_TAG}-{short_id(fixed_id)}"


def find_modality(case_dir: Path, suffix: str) -> Optional[Path]:
    """The single ``*-<suffix>.nii.gz`` in case_dir, or None."""
    hits = sorted(case_dir.glob(f"*-{suffix}{FILE_EXT}"))
    return hits[0] if hits else None


def discover_cases(data_dirs: Sequence[Path]) -> Dict[str, Path]:
    """Map case_id -> case dir for every complete case across the given dirs.

    A case counts only if it has all four modalities and a segmentation — a synthetic
    pair needs every one of them. The first occurrence of a duplicated case id wins.
    """
    cases: Dict[str, Path] = {}
    incomplete: List[str] = []
    for data_dir in data_dirs:
        if not data_dir.is_dir():
            raise FileNotFoundError(f"data dir not found: {data_dir}")
        for child in sorted(data_dir.iterdir()):
            if not child.is_dir() or not CASE_RE.match(child.name):
                continue
            if child.name in cases:
                continue
            needed = list(MODALITY_SUFFIXES) + [LABEL_SUFFIX]
            if any(find_modality(child, s) is None for s in needed):
                incomplete.append(child.name)
                continue
            cases[child.name] = child
    if incomplete:
        console.print(
            f"[yellow]{len(incomplete)} case(s) skipped — missing a modality or seg "
            f"(e.g. {incomplete[0]}).[/yellow]"
        )
    return cases


def test_patients_from_config(config_path: Path) -> Set[str]:
    """Patient ids in the config's seed-42 held-out test split.

    Reuses the repo's own splitter so the exclusion is exactly the partition every
    experiment holds out — no re-derivation, no drift.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.nnunet_replica.splits import holdout_split_ids
    from src.utils.experiment import load_config

    _, _, test_ids = holdout_split_ids(load_config(str(config_path)))
    return {patient_of(cid) for cid in test_ids}


def build_pairs(
    case_ids: Sequence[str],
    k: int,
    seed: int,
    max_pairs: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Pair each case (as MOVING) with k randomly chosen cases of OTHER patients.

    Deterministic given (sorted case_ids, k, seed). A case is never paired with itself,
    nor with another timepoint of the same patient — those would be within-patient
    registrations, which is not the augmentation we are after and would produce a
    near-duplicate of an existing case.

    ``max_pairs`` truncates a shuffled copy of the full list, so a capped run is an
    unbiased subset rather than "the first N case ids".
    """
    rng = random.Random(seed)
    ordered = sorted(case_ids)
    pairs: List[Tuple[str, str]] = []
    for moving in ordered:
        candidates = [c for c in ordered if patient_of(c) != patient_of(moving)]
        if not candidates:
            continue
        n = min(k, len(candidates))
        for fixed in rng.sample(candidates, n):
            pairs.append((moving, fixed))

    if max_pairs is not None and max_pairs < len(pairs):
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        pairs = sorted(shuffled[:max_pairs])
    return pairs


# =====================================================================================
# Geometry: ANTsPy <-> nibabel
# =====================================================================================

def ants_affine(img) -> np.ndarray:
    """The nibabel-style RAS affine implied by an ANTsImage's LPS direction/spacing/origin."""
    direction = np.asarray(img.direction, dtype=np.float64).reshape(3, 3)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction @ np.diag(np.asarray(img.spacing, dtype=np.float64))
    affine[:3, 3] = np.asarray(img.origin, dtype=np.float64)
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine   # LPS -> RAS


def assert_geometry_agreement(ants_module, path: Path) -> None:
    """Fail loudly if ANTsPy and nibabel disagree about a volume's array or affine.

    Outputs are written with nibabel (to preserve the fixed image's exact header) but
    the voxels come out of ANTs. That is only valid if the two libraries index the
    volume identically — this turns a silent axis transposition into an error, once,
    before thousands of CPU-hours are spent.
    """
    a = ants_module.image_read(str(path))
    n = nib.load(str(path))
    if tuple(a.shape) != tuple(n.shape):
        raise RuntimeError(f"ANTs/nibabel shape mismatch on {path}: {a.shape} vs {n.shape}")
    if not np.allclose(a.numpy(), n.get_fdata(), rtol=0, atol=1e-4):
        raise RuntimeError(
            f"ANTs and nibabel return different arrays for {path} — likely an axis "
            "transposition in this ANTsPy build. Refusing to write mis-oriented data."
        )
    if not np.allclose(ants_affine(a), n.affine, rtol=0, atol=1e-4):
        raise RuntimeError(
            f"ANTs and nibabel disagree on the affine for {path}:\n"
            f"{ants_affine(a)}\nvs\n{n.affine}"
        )


def output_dtype(source_dtype) -> np.dtype:
    """The dtype a warped modality is stored as.

    Interpolation produces continuous values, so a float source stays float — but
    capped at float32: some BraTS cases ship float64 volumes, and carrying that through
    doubles the size of the synthetic set for precision the pipeline immediately throws
    away (the preprocessed cache is float16/float32). Integer sources keep their own
    dtype, so a release that stores images as int16 round-trips unchanged.
    """
    dtype = np.dtype(source_dtype)
    if dtype.kind == "f":
        return np.dtype(np.float32) if dtype.itemsize > 4 else dtype
    return dtype


def write_nifti(array: np.ndarray, ref: nib.Nifti1Image, out_path: Path, dtype) -> None:
    """Write ``array`` on ``ref``'s grid, keeping ref's affine and header."""
    header = ref.header.copy()
    header.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(array, dtype=dtype), ref.affine, header)
    img.header.set_slope_inter(1, 0)   # drop any scaling carried by the reference header
    nib.save(img, str(out_path))


# =====================================================================================
# The registration itself
# =====================================================================================

def _drop_transform_files(reg: Dict) -> None:
    """ANTs writes its transforms to the system temp dir; thousands of pairs fill it."""
    for key in ("fwdtransforms", "invtransforms"):
        for path in reg.get(key) or []:
            try:
                os.remove(path)
            except OSError:
                pass


def check_warped_label(warped: np.ndarray, source_values: Set[int], name: str) -> np.ndarray:
    """Round the warped label to integers and verify no class was invented.

    ``genericLabel`` is supposed to return only values present in the input; this proves
    it did. Anything else — a fractional value, a class the source did not have, a class
    outside the scheme — means the label went through a continuous interpolator and the
    case must not be written.
    """
    rounded = np.rint(warped)
    if not np.allclose(warped, rounded, rtol=0, atol=1e-4):
        bad = warped[~np.isclose(warped, rounded, rtol=0, atol=1e-4)]
        raise ValueError(
            f"{name}: warped label has non-integer values (e.g. {bad[:5].tolist()}) — "
            "the label was interpolated continuously; refusing to write."
        )
    values = {int(v) for v in np.unique(rounded)}
    if not values <= VALID_LABELS:
        raise ValueError(
            f"{name}: warped label contains {sorted(values - VALID_LABELS)}, which are "
            f"outside the label scheme {sorted(VALID_LABELS)}."
        )
    if not values <= source_values:
        raise ValueError(
            f"{name}: warped label invented class(es) {sorted(values - source_values)} "
            f"not present in the source segmentation {sorted(source_values)}."
        )
    return rounded.astype(LABEL_DTYPE)


def register_pair(
    moving_dir: Path,
    fixed_dir: Path,
    out_dir: Path,
    transform: str,
    seed: int,
    with_subtraction: bool,
) -> Dict:
    """Warp every modality + the label of `moving_dir` into `fixed_dir`'s space.

    Returns a manifest entry. Writes into a temp directory and renames on success, so an
    interrupted run never leaves a half-written case that the resume logic would trust.
    """
    ants = load_ants()

    moving_id, fixed_id = moving_dir.name, fixed_dir.name
    case_name = synthetic_case_name(moving_id, fixed_id)

    fixed_ref_path = find_modality(fixed_dir, REGISTRATION_MODALITY)
    moving_ref_path = find_modality(moving_dir, REGISTRATION_MODALITY)
    if fixed_ref_path is None or moving_ref_path is None:
        raise FileNotFoundError(f"{case_name}: missing '{REGISTRATION_MODALITY}' on one side")

    fixed_img = ants.image_read(str(fixed_ref_path))
    moving_img = ants.image_read(str(moving_ref_path))

    # ---- ONE registration, on t1n only -------------------------------------------
    reg = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform=transform,
        random_seed=seed,
        verbose=False,
    )
    fwd = list(reg.get("fwdtransforms") or [])
    if not fwd:
        _drop_transform_files(reg)
        raise RuntimeError(f"{case_name}: ANTs returned no forward transform")

    # Header/affine of every output = the FIXED image's, because that is the space
    # everything has just been resampled into.
    fixed_ref_nib = nib.load(str(fixed_ref_path))

    staging = out_dir / f".tmp-{case_name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        warped_arrays: Dict[str, np.ndarray] = {}
        for suffix in MODALITY_SUFFIXES:
            src_path = find_modality(moving_dir, suffix)
            if src_path is None:
                raise FileNotFoundError(f"{case_name}: moving case has no '{suffix}'")
            warped = ants.apply_transforms(
                fixed=fixed_img,
                moving=ants.image_read(str(src_path)),
                transformlist=fwd,
                interpolator="linear",          # continuous data
            )
            array = warped.numpy()
            warped_arrays[suffix] = array
            write_nifti(array, fixed_ref_nib, staging / f"{case_name}-{suffix}{FILE_EXT}",
                        output_dtype(nib.load(str(src_path)).get_data_dtype()))

        # ---- the label: never linear ---------------------------------------------
        seg_path = find_modality(moving_dir, LABEL_SUFFIX)
        if seg_path is None:
            raise FileNotFoundError(f"{case_name}: moving case has no '{LABEL_SUFFIX}'")
        seg_img = ants.image_read(str(seg_path))
        source_values = {int(v) for v in np.unique(np.rint(seg_img.numpy()))}
        if not source_values <= VALID_LABELS:
            raise ValueError(
                f"{case_name}: SOURCE segmentation {seg_path.name} already contains "
                f"{sorted(source_values - VALID_LABELS)}, outside {sorted(VALID_LABELS)}."
            )
        warped_seg = ants.apply_transforms(
            fixed=fixed_img,
            moving=seg_img,
            transformlist=fwd,
            interpolator="genericLabel",        # multi-label, no new values
        )
        label = check_warped_label(warped_seg.numpy(), source_values, case_name)
        write_nifti(label, fixed_ref_nib, staging / f"{case_name}-{LABEL_SUFFIX}{FILE_EXT}",
                    LABEL_DTYPE)

        if with_subtraction:
            from precompute_subtraction import compute_subtraction
            sub = compute_subtraction(warped_arrays["t1c"], warped_arrays["t1n"], raw=False)
            write_nifti(sub, fixed_ref_nib,
                        staging / f"{case_name}-{SUBTRACTION_SUFFIX}{FILE_EXT}", np.float32)

        final = out_dir / case_name
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        _drop_transform_files(reg)

    return {
        "case_id": case_name,
        "moving_case": moving_id,
        "fixed_case": fixed_id,
        "moving_patient": patient_of(moving_id),
        "fixed_patient": patient_of(fixed_id),
        "transform": transform,
        "seed": seed,
        "labels_present": sorted(int(v) for v in np.unique(label)),
    }


def _entry_from_case_name(case_name: str) -> Optional[Dict]:
    """Rebuild a manifest entry from a synthetic case directory name.

    Both source patients are encoded in the name, so provenance survives even when the
    run that produced the case never got to write a manifest. The fields that are NOT
    in the name (transform, seed) are absent, and the entry says where it came from.
    """
    m = SYNTHETIC_CASE_RE.match(case_name)
    if not m:
        return None
    return {
        "case_id": case_name,
        "moving_case": f"BraTS-GLI-{m.group(1)}",
        "fixed_case": f"BraTS-GLI-{m.group(3)}",
        "moving_patient": m.group(2),
        "fixed_patient": m.group(4),
        "recovered_from_disk": True,
    }


def case_is_complete(out_dir: Path, case_name: str, with_subtraction: bool) -> bool:
    case_dir = out_dir / case_name
    if not case_dir.is_dir():
        return False
    suffixes = list(MODALITY_SUFFIXES) + [LABEL_SUFFIX]
    if with_subtraction:
        suffixes.append(SUBTRACTION_SUFFIX)
    return all((case_dir / f"{case_name}-{s}{FILE_EXT}").exists() for s in suffixes)


# =====================================================================================
# Worker entry point
# =====================================================================================

def _run_one(job: Dict) -> Tuple[str, Optional[Dict], Optional[str]]:
    # main() already exported this before importing ants, which is what actually takes
    # effect under fork; repeated here so a spawn-based start method (Windows/macOS),
    # where the child re-imports from scratch, is capped too.
    if job["ants_threads"] > 0:
        os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(job["ants_threads"]))
    case_name = job["case_name"]
    try:
        entry = register_pair(
            moving_dir=Path(job["moving_dir"]),
            fixed_dir=Path(job["fixed_dir"]),
            out_dir=Path(job["out_dir"]),
            transform=job["transform"],
            seed=job["seed"],
            with_subtraction=job["with_subtraction"],
        )
        return case_name, entry, None
    except Exception:
        return case_name, None, traceback.format_exc()


# =====================================================================================
# CLI
# =====================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-patient registration augmentation for BraTS GLI (Ferreira et al. 2023).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", action="append", required=True, type=Path,
                    help="BraTS case-directory root. Repeat to pool several releases.")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Where synthetic cases are written. Keep this SEPARATE from the "
                         "real data dirs.")
    ap.add_argument("-k", "--pairs-per-case", type=int, default=2,
                    help="How many other cases each case is warped into.")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="Cap the total number of pairs (unbiased random subset).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for pair selection and for ANTs' own RNG.")
    ap.add_argument("--transform", default="SyN", choices=["SyN", "SyNRA", "Affine"],
                    help="ANTs transform type. SyN = the real thing; SyNRA is a faster "
                         "rigid+affine+SyN; Affine is for quick smoke tests only.")
    ap.add_argument("--config", type=Path, default=None,
                    help="Experiment YAML. When given, patients in its seed-42 held-out "
                         "test split are excluded from pairing in BOTH roles.")
    ap.add_argument("--with-subtraction", action="store_true",
                    help=f"Also write the 5th '{SUBTRACTION_SUFFIX}' channel (t1c-t1n), "
                         "computed from the warped volumes, for 5-channel experiments.")
    ap.add_argument("--num-workers", type=int, default=4, help="Parallel pairs.")
    ap.add_argument("--ants-threads", type=int, default=0,
                    help="ITK threads per worker; 0 leaves ITK at its default (all cores). "
                         "0 measured ~2.5x faster overall than pinning 4 threads per "
                         "worker — the workers' serial phases interleave, so letting them "
                         "oversubscribe wins. Set it only to be a good neighbour on a "
                         "shared box.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate pairs that are already complete on disk.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the pairing plan and exit without registering anything.")
    args = ap.parse_args()

    cases = discover_cases(args.data_dir)
    if not cases:
        console.print("[red]No complete BraTS-GLI cases found — check --data-dir.[/red]")
        raise SystemExit(1)

    excluded: Set[str] = set()
    if args.config is not None:
        excluded = test_patients_from_config(args.config)
        cases = {cid: d for cid, d in cases.items() if patient_of(cid) not in excluded}
        console.print(
            f"[dim]--config {args.config.name}: excluded {len(excluded)} held-out test "
            f"patient(s) from pairing; {len(cases)} cases remain.[/dim]"
        )

    pairs = build_pairs(list(cases), args.pairs_per_case, args.seed, args.max_pairs)

    table = Table(title="Registration augmentation plan", style="bold cyan")
    table.add_column("", style="bold")
    table.add_column("", justify="right")
    table.add_row("source cases", str(len(cases)))
    table.add_row("patients", str(len({patient_of(c) for c in cases})))
    table.add_row("pairs per case (k)", str(args.pairs_per_case))
    table.add_row("pairs planned", str(len(pairs)))
    table.add_row("transform", args.transform)
    table.add_row("seed", str(args.seed))
    table.add_row("output", str(args.output_dir))
    console.print(table)

    if args.dry_run:
        for moving, fixed in pairs[:20]:
            console.print(f"  {synthetic_case_name(moving, fixed)}   "
                          f"[dim]{moving} -> {fixed}[/dim]")
        if len(pairs) > 20:
            console.print(f"  [dim]... and {len(pairs) - 20} more[/dim]")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    skipped = 0
    for index, (moving, fixed) in enumerate(pairs):
        case_name = synthetic_case_name(moving, fixed)
        if not args.overwrite and case_is_complete(args.output_dir, case_name,
                                                   args.with_subtraction):
            skipped += 1
            continue
        jobs.append({
            "case_name": case_name,
            "moving_dir": str(cases[moving]),
            "fixed_dir": str(cases[fixed]),
            "out_dir": str(args.output_dir),
            "transform": args.transform,
            # Per-pair seed: reproducible, but not the same ANTs RNG stream every pair.
            "seed": args.seed + index,
            "with_subtraction": args.with_subtraction,
            "ants_threads": args.ants_threads,
        })

    console.print(f"[bold]{len(pairs)} pairs, {skipped} already done, "
                  f"{len(jobs)} to register[/bold]")
    if not jobs:
        console.print("[green]Nothing to do.[/green]")
        return

    # Must be set BEFORE ants is first imported anywhere: ITK reads it when it initialises
    # its thread pool, and the fork'd workers inherit that already-initialised state, so
    # setting it from inside a worker is too late to have any effect. 0 = leave ITK alone,
    # which measured fastest here — see --ants-threads.
    if args.ants_threads > 0:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.ants_threads)

    # Fail fast, in the parent, before spawning hours of work: ANTsPy must be importable
    # and must agree with nibabel about how a volume is laid out.
    assert_geometry_agreement(
        load_ants(), find_modality(Path(jobs[0]["fixed_dir"]), REGISTRATION_MODALITY))
    console.print("[dim]ANTs/nibabel geometry check passed.[/dim]")

    entries: List[Dict] = []
    failures: List[Tuple[str, str]] = []
    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                  TimeRemainingColumn(), console=console) as progress:
        task = progress.add_task("registering", total=len(jobs))
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(_run_one, job) for job in jobs]
            for future in as_completed(futures):
                case_name, entry, err = future.result()
                if err:
                    failures.append((case_name, err))
                    console.print(f"[red]FAILED {case_name}[/red]")
                else:
                    entries.append(entry)
                progress.update(task, advance=1)

    _write_manifest(args, cases, excluded, entries)

    console.print(f"[bold green]{len(entries)} synthetic case(s) written to "
                  f"{args.output_dir}[/bold green]")
    if failures:
        console.print(f"[red bold]{len(failures)} pair(s) failed:[/red bold]")
        for case_name, err in failures[:3]:
            console.print(f"[red]{case_name}[/red]\n{err}")
        raise SystemExit(1)


def _write_manifest(args, cases: Dict[str, Path], excluded: Set[str],
                    entries: List[Dict]) -> None:
    """Rewrite the output dir's manifest (provenance for audits).

    Merges three sources so the manifest describes everything on disk, not just this
    run: the previous manifest, the cases this run produced, and — for a run that was
    killed before it could write a manifest — whatever complete case directories are
    present but unaccounted for, reconstructed from their names.
    """
    manifest_path = args.output_dir / MANIFEST_NAME
    existing: Dict[str, Dict] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                for entry in json.load(f).get("cases", []):
                    existing[entry["case_id"]] = entry
        except (OSError, ValueError, KeyError):
            console.print(f"[yellow]Could not read {manifest_path}; rewriting it.[/yellow]")
    for entry in entries:
        existing[entry["case_id"]] = entry

    recovered = 0
    for child in sorted(args.output_dir.iterdir()):
        if not child.is_dir() or child.name in existing or child.name.startswith("."):
            continue
        entry = _entry_from_case_name(child.name)
        if entry is None or not case_is_complete(args.output_dir, child.name,
                                                 args.with_subtraction):
            continue
        existing[child.name] = entry
        recovered += 1
    if recovered:
        console.print(f"[dim]Recovered {recovered} case(s) from disk that a previous "
                      f"interrupted run never recorded.[/dim]")

    payload = {
        "params": {
            "data_dirs": [str(d) for d in args.data_dir],
            "pairs_per_case": args.pairs_per_case,
            "max_pairs": args.max_pairs,
            "seed": args.seed,
            "transform": args.transform,
            "registration_modality": REGISTRATION_MODALITY,
            "modalities": list(MODALITY_SUFFIXES),
            "with_subtraction": args.with_subtraction,
            "config": str(args.config) if args.config else None,
            "excluded_test_patients": sorted(excluded),
            "source_cases": len(cases),
        },
        "cases": [existing[k] for k in sorted(existing)],
    }
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
