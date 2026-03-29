#!/usr/bin/env bash
# ==============================================================================
# Native nnU-Net v2 pipeline runner
#
# This script runs the full nnU-Net v2 pipeline:
#   1. Dataset conversion (BraTS -> nnU-Net format)
#   2. Experiment planning & preprocessing
#   3. Training (single fold using our patient-level split)
#   4. Inference on test set
#
# Usage:
#   bash nnunet_native/run_nnunet.sh [--data_dir PATH] [--gpu 0] [--config 3d_fullres]
#
# The patient-level split is enforced via a custom splits_final.json generated
# during conversion, ensuring no longitudinal leakage.
# ==============================================================================

set -euo pipefail

# Defaults
DATA_DIR="../Brats2024/training_data1_v2"
OUTPUT_DIR="./nnunet_data"
DATASET_ID=101
DATASET_NAME="BraTS2024"
GPU_ID=0
NNUNET_CONFIG="3d_fullres"    # 2d, 3d_fullres, or 3d_lowres
FOLD=0                         # We use fold 0 = our custom split
TRAINER="nnUNetTrainer"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)   DATA_DIR="$2"; shift 2;;
        --output_dir) OUTPUT_DIR="$2"; shift 2;;
        --gpu)        GPU_ID="$2"; shift 2;;
        --config)     NNUNET_CONFIG="$2"; shift 2;;
        --trainer)    TRAINER="$2"; shift 2;;
        *)            echo "Unknown arg: $1"; exit 1;;
    esac
done

DATASET_TAG="Dataset${DATASET_ID}_${DATASET_NAME}"

# Set nnU-Net environment
export nnUNet_raw="${OUTPUT_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${OUTPUT_DIR}/nnUNet_preprocessed"
export nnUNet_results="${OUTPUT_DIR}/nnUNet_results"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================"
echo " nnU-Net v2 Native Pipeline"
echo "============================================"
echo " Dataset:  ${DATASET_TAG}"
echo " Config:   ${NNUNET_CONFIG}"
echo " Trainer:  ${TRAINER}"
echo " GPU:      ${GPU_ID}"
echo " Raw:      ${nnUNet_raw}"
echo "============================================"

# ---- Step 1: Convert BraTS -> nnU-Net format ----
echo ""
echo "[Step 1/4] Converting BraTS data to nnU-Net format..."
python3 nnunet_native/convert_to_nnunet.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_name "${DATASET_NAME}"

# ---- Step 2: Plan & Preprocess ----
echo ""
echo "[Step 2/4] Planning and preprocessing..."
nnUNetv2_plan_and_preprocess \
    -d "${DATASET_ID}" \
    --verify_dataset_integrity \
    -c "${NNUNET_CONFIG}" \
    --verbose

# Copy our custom split into preprocessed dir (nnU-Net may overwrite during planning)
cp "${nnUNet_raw}/${DATASET_TAG}/splits_final.json" \
   "${nnUNet_preprocessed}/${DATASET_TAG}/splits_final.json" 2>/dev/null || true

# ---- Step 3: Train ----
echo ""
echo "[Step 3/4] Training fold ${FOLD}..."
nnUNetv2_train \
    "${DATASET_ID}" \
    "${NNUNET_CONFIG}" \
    "${FOLD}" \
    -tr "${TRAINER}"

# ---- Step 4: Predict on test set ----
echo ""
echo "[Step 4/4] Running inference on test set..."
PRED_DIR="${nnUNet_results}/${DATASET_TAG}/${TRAINER}__nnUNetPlans__${NNUNET_CONFIG}/fold_${FOLD}/test_predictions"
mkdir -p "${PRED_DIR}"

nnUNetv2_predict \
    -i "${nnUNet_raw}/${DATASET_TAG}/imagesTs" \
    -o "${PRED_DIR}" \
    -d "${DATASET_ID}" \
    -c "${NNUNET_CONFIG}" \
    -f "${FOLD}" \
    -tr "${TRAINER}"

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Predictions: ${PRED_DIR}"
echo " Run evaluation with:"
echo "   python3 nnunet_native/evaluate_nnunet.py \\"
echo "     --pred_dir ${PRED_DIR} \\"
echo "     --data_dir ${DATA_DIR}"
echo "============================================"
