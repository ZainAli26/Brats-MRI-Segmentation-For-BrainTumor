# Experiment Plan

## Hardware

- GPU: NVIDIA RTX 3070 Laptop (8 GB VRAM)
- Batch size constrained: bs=1 for all models at 128^3, grad_accum=2 for effective bs=2

## Overview

Seven phases, each answering a specific question. Run them in order —
later phases build on findings from earlier ones.

```
Phase 1: Baseline model comparison          (4 runs)   ~  4 days
Phase 2: Loss function study                (2 runs)   ~  2 days
Phase 3: Input resolution study             (2 runs)   ~  2 days
Phase 4: nnU-Net v2 architecture variants   (2 runs)   ~  2 days
Phase 5: Native nnU-Net v2 pipeline         (1 run)    ~  2 days
Phase 6: BraTS 2023 best model              (1 run)    ~  1 day
Phase 7: Cross-dataset generalization       (2 runs)   ~  2 days
                                           --------   ---------
                                           14 runs    ~ 15 days
```

---

## Phase 1 — Baseline Model Comparison

**Question:** Which architecture performs best on BraTS 2024 with default settings?

| Run | Config | Command |
|-----|--------|---------|
| 1.1 | `exp01_segresnet_baseline.yaml` | `python train.py --config experiments/exp01_segresnet_baseline.yaml` |
| 1.2 | `exp02_dynunet_baseline.yaml` | `python train.py --config experiments/exp02_dynunet_baseline.yaml` |
| 1.3 | `exp03_nnunet_v2_baseline.yaml` | `python train.py --config experiments/exp03_nnunet_v2_baseline.yaml` |
| 1.4 | `exp04_swin_unetr_baseline.yaml` | `python train.py --config experiments/exp04_swin_unetr_baseline.yaml` |

**After Phase 1:**
```bash
# Evaluate all four
for d in runs/segresnet_* runs/dynunet_* runs/nnunet_v2_* runs/swin_unetr_*; do
    python evaluate.py --run_dir "$d" --visualize_failures
done

# Compare side by side
python analyze_failures.py --run_dirs runs/segresnet_* runs/dynunet_* runs/nnunet_v2_* runs/swin_unetr_* --compare
```

**Decision:** Identify the top-2 models for further experiments.

---

## Phase 2 — Loss Function Study

**Question:** Does DiceFocal loss improve ET segmentation (small, hard-to-segment regions) vs DiceCE?

Uses the best model from Phase 1 (configs default to SegResNet; swap if needed).

| Run | Config | Command |
|-----|--------|---------|
| 2.1 | `exp05_dice_ce.yaml` | `python train.py --config experiments/exp05_dice_ce.yaml` |
| 2.2 | `exp06_dice_focal.yaml` | `python train.py --config experiments/exp06_dice_focal.yaml` |

**What to look at:** Compare ET Dice specifically. DiceFocal should help if
the model struggles with small/fragmented enhancing tumor.

---

## Phase 3 — Input Resolution Study

**Question:** Does larger crop size improve segmentation quality enough to justify
the slower training?

| Run | Config | Command |
|-----|--------|---------|
| 3.1 | `exp07_crop96.yaml` | `python train.py --config experiments/exp07_crop96.yaml` |
| 3.2 | `exp08_crop160.yaml` | `python train.py --config experiments/exp08_crop160.yaml` |

Compare against Phase 1 baseline at 128^3. The 96^3 run is a speed check —
if quality barely drops, use it for faster iteration.

---

## Phase 4 — nnU-Net v2 Architecture Variants

**Question:** Plain vs Residual encoder — which gives better Dice for BraTS?

| Run | Config | Command |
|-----|--------|---------|
| 4.1 | `exp09_nnunet_v2_plain.yaml` | `python train.py --config experiments/exp09_nnunet_v2_plain.yaml` |
| 4.2 | `exp10_nnunet_v2_residual.yaml` | `python train.py --config experiments/exp10_nnunet_v2_residual.yaml` |

---

## Phase 5 — Native nnU-Net v2 Pipeline

**Question:** How does nnU-Net v2 with its own self-configured preprocessing compare
to our fixed pipeline?

| Run | Config | Command |
|-----|--------|---------|
| 5.1 | (native) | `bash nnunet_native/run_nnunet.sh --data_dir ../Brats2024/training_data1_v2 --gpu 0` |

**Evaluate:**
```bash
python nnunet_native/evaluate_nnunet.py \
    --pred_dir nnunet_data/nnUNet_results/Dataset101_BraTS2024/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/test_predictions

# Compare native vs custom-loop nnU-Net v2
python analyze_failures.py \
    --run_dirs runs/nnunet_native_eval runs/nnunet_v2_* --compare
```

---

## Phase 5b — Native nnU-Net v2 ResEnc, 5-Fold CV (exp18 replica)

**Question:** Run the *exact* exp18 comparison (residual-encoder nnU-Net, 5-fold
patient-level CV on BraTS 2024) through the **real nnU-Net v2 framework** — its own
self-configured planning, preprocessing, 1000-epoch `nnUNetTrainer`, augmentation, and
sliding-window inference. How much of exp18's result is the architecture vs. our custom
training loop?

What is held fixed vs. exp18: dataset, residual-encoder family, patient-level 5-fold,
`split_seed=42`, label remap, and the ET/TC/WT metrics. What nnU-Net decides for itself:
patch size, normalization, batch size, ResEnc depth (ResEnc-L preset), epochs, LR schedule,
optimizer, loss, and augmentation.

**Data hygiene:** a test set (the same seed-42 10% patients as `create_patient_splits`) is
**held out of every training fold** — it goes to `imagesTs` and is never trained on. The
5-fold CV runs on the train+val pool only. Two numbers are reported: (a) out-of-fold
validation Dice (the CV metric, pool only) and (b) held-out test Dice from the 5-fold
ensemble (fully leak-free, comparable to the holdout native run and custom models on the
same test set). This differs from exp18, which K-folds *all* data with no held-out test.

| Run | Config (reference) | Command |
|-----|--------------------|---------|
| 5b.1 | `exp19_nnunet_native_resenc_5fold.yaml` | `bash nnunet_native/run_resenc_5fold.sh --gpu 0 --preset L` |

Designed for a 24 GB+ GPU (ResEnc-L). On an 8 GB card pass `--preset M`. Train a single
fold first with `--folds 0` to sanity-check before launching all five.

**Evaluate** (out-of-fold validation predictions = the true 5-fold CV metric — done
automatically as Step 4 of the runner, or manually):
```bash
python nnunet_native/evaluate_nnunet_kfold.py \
    --results_dir nnunet_data/nnUNet_results/Dataset102_BraTS2024ResEnc/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres \
    --data_dir ../Brats2024/training_data1_v2 \
    --output_dir runs/exp19_nnunet_native_resenc_eval

# Head-to-head: real framework (exp19) vs custom loop (exp18)
python analyze_failures.py \
    --run_dirs runs/exp19_nnunet_native_resenc_eval runs/<exp18_eval_dir> --compare
```

---

## Phase 6 — BraTS 2023 Best Model

**Question:** Does the best architecture from Phase 1 also perform well on BraTS 2023?

| Run | Config | Command |
|-----|--------|---------|
| 6.1 | `exp11_brats2023_best.yaml` | `python train.py --config experiments/exp11_brats2023_best.yaml` |

---

## Phase 7 — Cross-Dataset Generalization

**Question:** Does a model trained on one year's data generalize to the other?

| Run | Config | Command |
|-----|--------|---------|
| 7.1 | `exp12_train2024_test2023.yaml` | `python train.py --config experiments/exp12_train2024_test2023.yaml` |
| 7.2 | `exp13_train2023_test2024.yaml` | `python train.py --config experiments/exp13_train2023_test2024.yaml` |

These train on one dataset and evaluate on the other's test split.
Requires manual evaluation step (see configs for instructions).

---

## After All Experiments

```bash
# Final comprehensive comparison across all runs
python analyze_failures.py \
    --run_dirs runs/exp*_* runs/nnunet_native_eval \
    --compare

# Identify common failure patterns
python analyze_failures.py --run_dir <best_run> --split test
```

## Key Metrics to Track

For each experiment, record in a spreadsheet or compare via `analyze_failures.py`:

| Metric | What it tells you |
|--------|-------------------|
| Dice WT | Overall tumor detection |
| Dice TC | Core tumor accuracy (NCR + ET) |
| Dice ET | Hardest subregion — small, fragmented |
| HD95 ET | Surface distance for ET — penalizes outlier errors |
| Small tumor ET Dice | Performance on cases with <500 ET voxels |
| Missed ET count | False negative rate for enhancing tumor |
