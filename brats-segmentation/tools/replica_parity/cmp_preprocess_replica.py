#!/usr/bin/env python3
"""Build the replica's case cache for Dataset500_ReplicaCmp.

Reads the *same files* the native run reads (Dataset500's imagesTr/labelsTr), through the
replica's own preprocessing, with the same 8 GB ResEnc-M plan. store_dtype is float32 so the
cache matches nnU-Net's preprocessed arrays byte for byte — the point of this run is to
isolate the training loop, so the one known deviation of exp20's default (float16 cache) is
removed here and measured separately.
"""
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path("/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/brats-segmentation")
# Durable work dir. This lived in the session scratchpad once; a reboot destroyed it
# along with the cache and every run. nnunet_data/ is gitignored, so heavy artefacts
# stay out of git while these scripts stay in it.
SP = Path("/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/"
          "brats-segmentation/nnunet_data/replica_parity")
sys.path.insert(0, str(REPO))

RAW = REPO / "nnunet_data/nnUNet_raw/Dataset500_ReplicaCmp"
PLANS = REPO / "plans/nnUNetResEncUNetMPlans_8G_cmp.json"
OUT = SP / "replica_cache/Dataset500_ReplicaCmp_fp32"  # .cache/ in the repo is root-owned (Docker)
NUM_CHANNELS = 4
FG_LABELS = [1, 2, 3, 4]
STORE_DTYPE = "float32"


def one(case_id):
    from src.nnunet_replica.plans import Plans
    from src.nnunet_replica.preprocessing import (
        case_is_cached, preprocess_case, save_preprocessed_case,
    )
    try:
        if case_is_cached(OUT, case_id):
            return case_id, None
        plans = Plans.load(PLANS)
        cfg = plans.get_configuration("3d_fullres")
        imgs = [str(RAW / "imagesTr" / f"{case_id}_{i:04d}.nii.gz") for i in range(NUM_CHANNELS)]
        data, seg, props = preprocess_case(
            imgs, str(RAW / "labelsTr" / f"{case_id}.nii.gz"), cfg, FG_LABELS,
            label_map=None, transpose_forward=plans.transpose_forward,
        )
        save_preprocessed_case(OUT, case_id, data, seg, props, STORE_DTYPE)
        return case_id, None
    except Exception:
        return case_id, traceback.format_exc()


def main():
    import json
    split = json.load(open(SP / "cmp_splits_fold0.json"))[0]
    cases = split["train"] + split["val"]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} cases -> {OUT} ({STORE_DTYPE})")

    fails = []
    done = 0
    with ProcessPoolExecutor(max_workers=12) as pool:
        futs = [pool.submit(one, c) for c in cases]
        for f in as_completed(futs):
            cid, err = f.result()
            done += 1
            if err:
                fails.append((cid, err))
                print(f"FAILED {cid}")
            if done % 25 == 0:
                print(f"  {done}/{len(cases)}", flush=True)
    if fails:
        print(fails[0][1])
        return 1
    print("cache complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

