#!/usr/bin/env bash
# ==============================================================================
# Exp 19 — Native nnU-Net v2 ResEnc-L, 5-fold CV  (real-framework replica of exp18)
#
# Swaps our custom k-fold training loop (exp18) for the REAL nnU-Net v2 framework:
#   1. Convert BraTS 2024 -> nnU-Net format. A test set (same seed-42 patients as
#      the holdout split) is HELD OUT in imagesTs and never trained on. The
#      remaining train+val patients go to imagesTr with 5 patient-level folds.
#   2. Plan & preprocess with the ResEnc-L planner (lets nnU-Net self-configure
#      patch size, normalization, batch size, and residual-encoder depth)
#   3. Train all 5 folds with the stock nnUNetTrainer (1000 epochs each)
#   4. Bridge the out-of-fold validation predictions (CV metric) into our shared
#      metrics so exp19 is comparable to exp18 via analyze_failures.py
#   5. Ensemble the trained folds and predict the HELD-OUT test set, then evaluate
#      it with the shared metrics — a fully leak-free test number.
#
# Usage:
#   bash nnunet_native/run_resenc_5fold.sh [--gpu 0] [--data_dir PATH] \
#        [--preset L] [--folds 0,1,2,3,4]
#
# Designed for a 24 GB+ GPU (ResEnc-L). For an 8 GB card use --preset M.
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- Defaults ----
DATA_DIR="../Brats2024/training_data1_v2"
OUTPUT_DIR="./nnunet_data"
DATASET_ID=102                      # distinct from Phase-5 holdout (101)
DATASET_NAME="BraTS2024ResEnc"
GPU_ID=0
NNUNET_CONFIG="3d_fullres"
N_FOLDS=5
FOLDS="0,1,2,3,4"                   # which folds to train this invocation
PRESET="L"                          # M | L | XL  -> ResEnc preset
TRAINER="nnUNetTrainer"

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)   DATA_DIR="$2"; shift 2;;
        --output_dir) OUTPUT_DIR="$2"; shift 2;;
        --gpu)        GPU_ID="$2"; shift 2;;
        --config)     NNUNET_CONFIG="$2"; shift 2;;
        --dataset_id) DATASET_ID="$2"; shift 2;;
        --n_folds)    N_FOLDS="$2"; shift 2;;
        --folds)      FOLDS="$2"; shift 2;;
        --preset)     PRESET="$2"; shift 2;;
        --trainer)    TRAINER="$2"; shift 2;;
        *)            echo "Unknown arg: $1"; exit 1;;
    esac
done

case "$PRESET" in
    M)  PLANNER="nnUNetPlannerResEncM";  PLANS="nnUNetResEncUNetMPlans";;
    L)  PLANNER="nnUNetPlannerResEncL";  PLANS="nnUNetResEncUNetLPlans";;
    XL) PLANNER="nnUNetPlannerResEncXL"; PLANS="nnUNetResEncUNetXLPlans";;
    *)  echo "Unknown preset: $PRESET (use M, L, or XL)"; exit 1;;
esac

DATASET_TAG="Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}"

export nnUNet_raw="${OUTPUT_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${OUTPUT_DIR}/nnUNet_preprocessed"
export nnUNet_results="${OUTPUT_DIR}/nnUNet_results"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================"
echo " Exp 19 — Native nnU-Net v2 ResEnc-${PRESET}, ${N_FOLDS}-fold CV"
echo "============================================"
echo " Dataset:  ${DATASET_TAG}"
echo " Config:   ${NNUNET_CONFIG}"
echo " Planner:  ${PLANNER}"
echo " Plans:    ${PLANS}"
echo " Trainer:  ${TRAINER}"
echo " Folds:    ${FOLDS}"
echo " GPU:      ${GPU_ID}"
echo "============================================"

# ---- Step 1: Convert BraTS -> nnU-Net (K-fold pool + held-out test) ----
echo ""
echo "[Step 1/5] Converting BraTS data (K-fold train+val, held-out test)..."
python3 nnunet_native/convert_to_nnunet.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_name "${DATASET_NAME}" \
    --mode kfold \
    --n_folds "${N_FOLDS}" \
    --split_seed 42

# ---- Step 2: Plan & Preprocess with the ResEnc planner ----
echo ""
echo "[Step 2/5] Planning & preprocessing with ${PLANNER}..."
nnUNetv2_plan_and_preprocess \
    -d "${DATASET_ID}" \
    -pl "${PLANNER}" \
    -c "${NNUNET_CONFIG}" \
    --verify_dataset_integrity \
    --verbose

# Re-assert our patient-level folds: planning regenerates splits_final.json with
# nnU-Net's own random KFold, so copy ours back over it.
cp "${nnUNet_raw}/${DATASET_TAG}/splits_final.json" \
   "${nnUNet_preprocessed}/${DATASET_TAG}/splits_final.json"
echo "  Restored patient-level splits_final.json (exp18 folds)."

# ---- Step 3: Train each fold (writes OOF validation preds via --val_best) ----
echo ""
echo "[Step 3/5] Training folds: ${FOLDS}"
IFS=',' read -ra FOLD_ARR <<< "${FOLDS}"
for FOLD in "${FOLD_ARR[@]}"; do
    echo ""
    echo "  >>> Fold ${FOLD} — $(date)"
    nnUNetv2_train \
        "${DATASET_ID}" \
        "${NNUNET_CONFIG}" \
        "${FOLD}" \
        -p "${PLANS}" \
        -tr "${TRAINER}" \
        --npz
done

# ---- Step 4: Bridge out-of-fold validation preds into our shared metrics ----
echo ""
echo "[Step 4/5] Evaluating out-of-fold validation predictions (CV metric)..."
RESULTS_SUBDIR="${nnUNet_results}/${DATASET_TAG}/${TRAINER}__${PLANS}__${NNUNET_CONFIG}"
python3 nnunet_native/evaluate_nnunet_kfold.py \
    --results_dir "${RESULTS_SUBDIR}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "runs/exp19_nnunet_native_resenc_eval" \
    --n_folds "${N_FOLDS}"

# ---- Step 5: Ensemble trained folds, predict HELD-OUT test set, evaluate ----
echo ""
echo "[Step 5/5] Predicting the held-out test set (ensemble of folds: ${FOLDS})..."
TEST_PRED_DIR="${RESULTS_SUBDIR}/test_predictions"
mkdir -p "${TEST_PRED_DIR}"
nnUNetv2_predict \
    -i "${nnUNet_raw}/${DATASET_TAG}/imagesTs" \
    -o "${TEST_PRED_DIR}" \
    -d "${DATASET_ID}" \
    -c "${NNUNET_CONFIG}" \
    -p "${PLANS}" \
    -tr "${TRAINER}" \
    -f "${FOLD_ARR[@]}"

echo " Evaluating held-out test predictions with shared metrics..."
python3 nnunet_native/evaluate_nnunet.py \
    --pred_dir "${TEST_PRED_DIR}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "runs/exp19_nnunet_native_resenc_test_eval"

echo ""
echo "============================================"
echo " Exp 19 complete!"
echo "  CV (out-of-fold) metric:  runs/exp19_nnunet_native_resenc_eval"
echo "  Held-out test metric:     runs/exp19_nnunet_native_resenc_test_eval"
echo ""
echo " The test set (seed-42 10%) was held out of every training fold."
echo ""
echo " Compare against exp18 (custom-loop residual 5-fold):"
echo "   python analyze_failures.py \\"
echo "     --run_dirs runs/exp19_nnunet_native_resenc_eval runs/<exp18_fold_or_eval_dir> \\"
echo "     --compare"
echo "============================================"
