#!/usr/bin/env python3
"""Apply the shipped small-component PP to a directory of BraTS label maps.

Thresholds (label: min voxels): 2:200, 3:100, 4:250 (label 1 untouched) — the
CV-determined, test-confirmed joint combo. Writes PP'd copies preserving geometry.

Usage: apply_joint_pp.py --in_dir <segs> --out_dir <pp_segs> [--workers N]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.nnunet_postprocessing import remove_small_components  # noqa: E402

JOINT = {2: 200, 3: 100, 4: 250}


def one(args):
    src, dst = args
    img = nib.load(str(src))
    seg = np.asanyarray(img.dataobj).astype(np.uint8)
    for lbl, thr in JOINT.items():
        if thr > 0:
            seg = remove_small_components(seg, [lbl], thr)
    nib.save(nib.Nifti1Image(seg, img.affine, img.header), str(dst))
    return dst.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, out / p.name) for p in sorted(Path(a.in_dir).glob("*.nii.gz"))
            if not (out / p.name).exists()]
    done_already = len(list(out.glob("*.nii.gz")))
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(one, jobs), 1):
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print(f"PP done: {len(jobs)} new + {done_already} existing -> {out}")


if __name__ == "__main__":
    main()
