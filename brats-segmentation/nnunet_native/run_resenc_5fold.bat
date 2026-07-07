@echo off
REM ============================================================================
REM  Exp 19 - Native nnU-Net v2 ResEnc, 5-fold CV  (Windows / cmd port)
REM
REM  Windows equivalent of run_resenc_5fold.sh. Runs the REAL nnU-Net v2
REM  framework end-to-end:
REM    1. Convert BraTS 2024 -> nnU-Net format. Pools the two training dirs
REM       (training_data1_v2 + training_data_additional) into one patient-level
REM       split; a seed-42 test set is HELD OUT in imagesTs and never trained on.
REM    2. Plan & preprocess with the ResEnc-M planner (~11 GB target -> fits 12 GB).
REM    3. Train the requested folds with the stock nnUNetTrainer.
REM    4. Bridge out-of-fold validation preds into the shared metrics (CV number).
REM    5. Ensemble folds, predict the held-out test set, evaluate it.
REM
REM  USAGE (from anywhere; the script cd's to the repo root itself):
REM      nnunet_native\run_resenc_5fold.bat
REM
REM  EDIT THE VARIABLES IN THE "CONFIG" BLOCK BELOW before the first run.
REM ============================================================================

setlocal enabledelayedexpansion

REM ---- Move to repo root (this script lives in <repo>\nnunet_native) ----
cd /d "%~dp0.."

REM ============================ CONFIG (edit me) ==============================
REM Data dirs to POOL. Both are passed to the converter and split at patient
REM level. Quote each because the path contains a space ("ROG STRIX").
set "DATA_DIR_1=C:\Users\ROG STRIX\Downloads\Zain\Brats-MRI-Segmentation-For-BrainTumor\Brats2024\training_data1_v2"
set "DATA_DIR_2=C:\Users\ROG STRIX\Downloads\Zain\Brats-MRI-Segmentation-For-BrainTumor\Brats2024\training_data_additional"

set "OUTPUT_DIR=%CD%\nnunet_data"
set "DATASET_ID=102"
set "DATASET_NAME=BraTS2024ResEnc"
set "GPU_ID=0"
set "NNUNET_CONFIG=3d_fullres"
set "N_FOLDS=5"

REM ResEnc-M planner, but planned for an explicit VRAM target (see GPU_MEM_TARGET).
REM Overriding the target writes a NEW plans file (PLANS), so the preset's default
REM plans stay untouched. Everything downstream trains/predicts with -p %PLANS%.
set "PLANNER=nnUNetPlannerResEncM"
set "GPU_MEM_TARGET=11"
set "PLANS=nnUNetResEncM_11GBPlans"
set "TRAINER=nnUNetTrainer"

REM Which folds to train, space-separated.
REM   Full experiment (faithful exp19):   0 1 2 3 4
REM   Quick first end-to-end test:         0
REM NOTE: each fold trains 1000 epochs -> MANY hours on a 12 GB GPU. Start with
REM a single fold (set "FOLDS=0") to confirm the pipeline before committing to 5.
set "FOLDS=0 1 2 3 4"
REM ===========================================================================

set "DATASET_TAG=Dataset%DATASET_ID%_%DATASET_NAME%"

REM ---- nnU-Net environment (inherited by every child process) ----
set "nnUNet_raw=%OUTPUT_DIR%\nnUNet_raw"
set "nnUNet_preprocessed=%OUTPUT_DIR%\nnUNet_preprocessed"
set "nnUNet_results=%OUTPUT_DIR%\nnUNet_results"
set "CUDA_VISIBLE_DEVICES=%GPU_ID%"

echo ============================================
echo  Exp 19 - Native nnU-Net v2 ResEnc (Windows)
echo ============================================
echo  Dataset:   %DATASET_TAG%
echo  Config:    %NNUNET_CONFIG%
echo  Planner:   %PLANNER%
echo  Plans:     %PLANS%
echo  Folds:     %FOLDS%
echo  GPU:       %GPU_ID%
echo  Raw:       %nnUNet_raw%
echo ============================================

REM ---- Step 1/5: Convert BraTS -> nnU-Net (pooled K-fold + held-out test) ----
echo.
echo [Step 1/5] Converting BraTS data (pooling both training dirs)...
python nnunet_native\convert_to_nnunet.py --data_dir "%DATA_DIR_1%" "%DATA_DIR_2%" --output_dir "%OUTPUT_DIR%" --dataset_id %DATASET_ID% --dataset_name %DATASET_NAME% --mode kfold --n_folds %N_FOLDS% --split_seed 42
if errorlevel 1 goto :error

REM ---- Step 2/5: Plan & preprocess for an explicit %GPU_MEM_TARGET% GB target ----
REM Split into fingerprint -> plan_experiment -> preprocess because
REM -gpu_memory_target / -overwrite_plans_name reliably live on plan_experiment.
echo.
echo [Step 2/5] Planning ^& preprocessing with %PLANNER% for %GPU_MEM_TARGET% GB...
nnUNetv2_extract_fingerprint -d %DATASET_ID% --verify_dataset_integrity
if errorlevel 1 goto :error
nnUNetv2_plan_experiment -d %DATASET_ID% -pl %PLANNER% -gpu_memory_target %GPU_MEM_TARGET% -overwrite_plans_name %PLANS%
if errorlevel 1 goto :error
nnUNetv2_preprocess -d %DATASET_ID% -plans_name %PLANS% -c %NNUNET_CONFIG%
if errorlevel 1 goto :error

REM Re-assert our patient-level folds: planning regenerates splits_final.json
REM with nnU-Net's own random KFold, so copy ours back over it.
copy /Y "%nnUNet_raw%\%DATASET_TAG%\splits_final.json" "%nnUNet_preprocessed%\%DATASET_TAG%\splits_final.json"
if errorlevel 1 goto :error
echo   Restored patient-level splits_final.json (exp18 folds).

REM ---- Step 3/5: Train each fold ----
echo.
echo [Step 3/5] Training folds: %FOLDS%
for %%F in (%FOLDS%) do (
    echo.
    echo   ^>^>^> Fold %%F
    nnUNetv2_train %DATASET_ID% %NNUNET_CONFIG% %%F -p %PLANS% -tr %TRAINER% --npz
    if errorlevel 1 goto :error
)

REM ---- Step 4/5: Bridge out-of-fold validation preds into shared metrics ----
echo.
echo [Step 4/5] Evaluating out-of-fold validation predictions (CV metric)...
set "RESULTS_SUBDIR=%nnUNet_results%\%DATASET_TAG%\%TRAINER%__%PLANS%__%NNUNET_CONFIG%"
python nnunet_native\evaluate_nnunet_kfold.py --results_dir "%RESULTS_SUBDIR%" --data_dir "%DATA_DIR_1%" "%DATA_DIR_2%" --output_dir "runs\exp19_nnunet_native_resenc_eval" --n_folds %N_FOLDS%
if errorlevel 1 goto :error

REM ---- Step 5/5: Ensemble folds, predict HELD-OUT test set, evaluate ----
echo.
echo [Step 5/5] Predicting the held-out test set (ensemble of folds: %FOLDS%)...
set "TEST_PRED_DIR=%RESULTS_SUBDIR%\test_predictions"
if not exist "%TEST_PRED_DIR%" mkdir "%TEST_PRED_DIR%"
nnUNetv2_predict -i "%nnUNet_raw%\%DATASET_TAG%\imagesTs" -o "%TEST_PRED_DIR%" -d %DATASET_ID% -c %NNUNET_CONFIG% -p %PLANS% -tr %TRAINER% -f %FOLDS%
if errorlevel 1 goto :error

echo   Evaluating held-out test predictions with shared metrics...
python nnunet_native\evaluate_nnunet.py --pred_dir "%TEST_PRED_DIR%" --data_dir "%DATA_DIR_1%" "%DATA_DIR_2%" --output_dir "runs\exp19_nnunet_native_resenc_test_eval"
if errorlevel 1 goto :error

echo.
echo ============================================
echo  Exp 19 complete!
echo   CV (out-of-fold) metric:  runs\exp19_nnunet_native_resenc_eval
echo   Held-out test metric:     runs\exp19_nnunet_native_resenc_test_eval
echo ============================================
goto :eof

:error
echo.
echo *** FAILED at the step above (exit code %errorlevel%). Fix it and re-run. ***
echo     The script is resumable: completed folds/preprocessing are cached and
echo     skipped, so re-running continues where it stopped.
exit /b %errorlevel%
