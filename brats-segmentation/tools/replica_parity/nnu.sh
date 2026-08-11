#!/bin/bash
# Run an nnU-Net v2.4.2 entry point against the comparison dataset.
#   nnu.sh <entrypoint-function> [args...]
# 2.4.2 lives as a source tree on PYTHONPATH (shadowing the installed 2.1) rather than as
# an install, so torch / batchgenerators / dynamic_network_architectures stay exactly the
# ones the replica imports — the comparison must not differ by library version.
set -euo pipefail

SP=/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/brats-segmentation/nnunet_data/replica_parity
BASE=/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/brats-segmentation/nnunet_data

export PYTHONPATH="$SP/nnUNet242"
export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/nnUNet_results"
export nnUNet_n_proc_DA=12
export OMP_NUM_THREADS=1

EP="$1"; shift
case "$EP" in
  fingerprint) MOD=nnunetv2.experiment_planning.plan_and_preprocess_entrypoints; FN=extract_fingerprint_entry ;;
  plan)        MOD=nnunetv2.experiment_planning.plan_and_preprocess_entrypoints; FN=plan_experiment_entry ;;
  preprocess)  MOD=nnunetv2.experiment_planning.plan_and_preprocess_entrypoints; FN=preprocess_entry ;;
  train)       MOD=nnunetv2.run.run_training;                                    FN=run_training_entry ;;
  predict)     MOD=nnunetv2.inference.predict_from_raw_data;                     FN=predict_entry_point ;;
  *) echo "unknown entrypoint: $EP" >&2; exit 2 ;;
esac

exec python3 -c "
import sys
sys.argv[0] = '$FN'
from $MOD import $FN as _f
_f()
" "$@"

