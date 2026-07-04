#!/usr/bin/env python3
"""Precompute the T1Gd−T1 (t1c − t1n) subtraction channel for BraTS cases.

Used by exp22 / exp23, which add a 5th input channel read from disk as
``*-sub.nii.gz``. The custom data pipeline finds modalities via
``glob("*-{mod}.nii.gz")`` (src/data/dataset.py), so this writes one
``<prefix>-sub.nii.gz`` per case directory, on the SAME grid/affine as the
source modalities.

BraTS modalities are already skull-stripped and co-registered to a common 1mm
isotropic grid, so t1c and t1n share an affine and shape — a direct voxel
subtraction is valid (no resampling). Raw t1c/t1n intensities, however, are on
arbitrary scales, so a raw difference reflects scale mismatch rather than
contrast enhancement. By default each volume is z-score normalized within the
brain mask (nonzero voxels) BEFORE subtracting; pass --raw for a plain t1c−t1n.

The output keeps background (outside the brain mask) at exactly 0 so the
training pipeline's NormalizeIntensity(nonzero=True) treats it like the other
modalities.

Usage (run from brats-segmentation/):
    python precompute_subtraction.py --data_dir ../Brats2024/training_data1_v2
    python precompute_subtraction.py --data_dir ../Brats2024/training_data1_v2 \
        --data_dir ../Brats2024/validation_data --overwrite
    python precompute_subtraction.py --data_dir <dir> --suffix sub --raw
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional — fall back to a no-op wrapper
    def tqdm(it, **kwargs):
        return it


def _find_one(case_dir: Path, mod: str):
    """Return the single file matching *-{mod}.nii.gz in case_dir, or None."""
    hits = sorted(case_dir.glob(f"*-{mod}.nii.gz"))
    return hits[0] if hits else None


def _zscore_nonzero(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score normalize within `mask`; voxels outside mask are left as-is (0)."""
    out = np.zeros_like(vol, dtype=np.float32)
    vals = vol[mask]
    if vals.size == 0:
        return out
    mean = vals.mean()
    std = vals.std()
    if std < 1e-8:
        std = 1.0
    out[mask] = (vals - mean) / std
    return out


def compute_subtraction(t1c: np.ndarray, t1n: np.ndarray, raw: bool) -> np.ndarray:
    """T1Gd−T1 map. Brain mask = voxels nonzero in BOTH t1c and t1n.

    Background (outside the mask) is forced to 0 so downstream nonzero
    normalization ignores it, matching the other modality channels.
    """
    t1c = t1c.astype(np.float32)
    t1n = t1n.astype(np.float32)
    mask = (t1c != 0) & (t1n != 0)
    if raw:
        sub = t1c - t1n
    else:
        sub = _zscore_nonzero(t1c, mask) - _zscore_nonzero(t1n, mask)
    sub[~mask] = 0.0
    return sub.astype(np.float32)


def process_case(case_dir: Path, suffix: str, raw: bool, overwrite: bool) -> str:
    """Returns one of: 'written', 'skipped-exists', 'skipped-missing', 'skipped-mismatch'."""
    t1c_path = _find_one(case_dir, "t1c")
    t1n_path = _find_one(case_dir, "t1n")
    if t1c_path is None or t1n_path is None:
        return "skipped-missing"

    # Output name mirrors the t1c file's prefix: BraTS-...-t1c.nii.gz -> BraTS-...-<suffix>.nii.gz
    prefix = t1c_path.name[: -len("t1c.nii.gz")]  # keeps the trailing '-'
    out_path = case_dir / f"{prefix}{suffix}.nii.gz"
    if out_path.exists() and not overwrite:
        return "skipped-exists"

    t1c_img = nib.load(str(t1c_path))
    t1n_img = nib.load(str(t1n_path))
    t1c = t1c_img.get_fdata()
    t1n = t1n_img.get_fdata()

    # Co-registration sanity: BraTS modalities must already share a grid.
    if t1c.shape != t1n.shape or not np.allclose(t1c_img.affine, t1n_img.affine, atol=1e-3):
        return "skipped-mismatch"

    sub = compute_subtraction(t1c, t1n, raw=raw)
    # Reuse t1c geometry; write float32 so the header dtype matches the data.
    out_img = nib.Nifti1Image(sub, t1c_img.affine, t1c_img.header)
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, str(out_path))
    return "written"


def iter_case_dirs(data_dir: Path):
    """Case dirs = immediate subdirectories that contain a t1c file."""
    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and _find_one(child, "t1c") is not None:
            yield child


def main():
    parser = argparse.ArgumentParser(
        description="Precompute T1Gd−T1 (t1c−t1n) subtraction channel for BraTS cases."
    )
    parser.add_argument(
        "--data_dir", action="append", required=True, type=Path,
        help="BraTS data directory (each subdir is a case). Repeat for multiple dirs.",
    )
    parser.add_argument(
        "--suffix", default="sub",
        help="Modality suffix to write as *-<suffix>.nii.gz (default: sub). "
             "Must match the 5th entry in the experiment's data.modalities list.",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Plain t1c−t1n on raw intensities (default: z-score each within the "
             "brain mask before subtracting).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute and overwrite existing *-<suffix>.nii.gz files.",
    )
    args = parser.parse_args()

    totals = {"written": 0, "skipped-exists": 0, "skipped-missing": 0, "skipped-mismatch": 0}
    for data_dir in args.data_dir:
        if not data_dir.is_dir():
            print(f"[!] not a directory, skipping: {data_dir}")
            continue
        cases = list(iter_case_dirs(data_dir))
        print(f"{data_dir}: {len(cases)} cases")
        for case_dir in tqdm(cases, desc=data_dir.name):
            status = process_case(case_dir, args.suffix, args.raw, args.overwrite)
            totals[status] += 1

    print("\nDone:")
    print(f"  written          : {totals['written']}")
    print(f"  skipped (exists) : {totals['skipped-exists']}  (use --overwrite to redo)")
    print(f"  skipped (missing t1c/t1n) : {totals['skipped-missing']}")
    print(f"  skipped (grid mismatch)   : {totals['skipped-mismatch']}")
    if totals["skipped-mismatch"]:
        print("  [!] grid-mismatch cases need co-registration before subtraction.")


if __name__ == "__main__":
    main()
