# BraTS Segmentation — Experiment Results (6 Experiments)

## Hardware & Setup

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA RTX 3070 Laptop (8 GB VRAM) |
| **Dataset** | BraTS 2024 GLI |
| **Total cases** | 1350 (613 unique patients) |
| **Split** | Patient-level 75/15/10 — 1021 train, 200 val, 129 test |
| **Split seed** | 42 (deterministic, reproducible) |
| **Evaluation** | Sliding window inference on 129 test cases |

## Common Configuration (Shared Across All Experiments)

| Parameter | Value |
|-----------|-------|
| Modalities | t1c, t1n, t2f, t2w (4-channel input) |
| Normalization | Z-score on nonzero voxels, per channel |
| Orientation | RAS |
| Voxel spacing | 1.0 × 1.0 × 1.0 mm (resampled) |
| Foreground crop | CropForeground + SpatialPad |
| Training crop | RandCropByPosNegLabeld (pos=3, neg=1) |
| Augmentation | Random flip (p=0.5, all axes), Random rotate90 (p=0.3), Intensity shift (±0.1), Intensity scale (±0.1) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| LR scheduler | Cosine warm restarts (T₀=50, T_mult=2, η_min=1e-6) |
| Batch size | 1 (gradient accumulation = 2, effective batch = 2) |
| Mixed precision | AMP (FP16) enabled |
| Max epochs | 300 |
| Val interval | Every 5 epochs |
| Early stopping | Patience = 30 val checks (150 epochs without improvement) |
| Sliding window overlap | 0.5 |
| Label remapping | BraTS 2024: {0→0, 1→1, 2→2, 4→3} for contiguous classes |

---

## Experiment 01 — SegResNet Baseline

**Question:** Baseline performance of SegResNet (BraTS 2018 challenge winner architecture).

| Parameter | Value |
|-----------|-------|
| **Model** | SegResNet (MONAI) |
| **Parameters** | 18.8M |
| **Crop size** | **128 × 128 × 128** |
| **Loss** | DiceCE (dice_weight=1.0, ce_weight=1.0) |
| **Architecture** | init_filters=32, blocks_down=[1,2,2,4], blocks_up=[1,1,1], dropout=0.2 |
| **SW batch size** | 2 |
| **Num workers** | 4 |

| Metric | Value |
|--------|-------|
| **Best epoch** | 285 / 300 |
| **Val Dice (per-class mean)** | 0.4635 |
| **Test Dice ET** | 0.5723 |
| **Test Dice TC** | 0.6029 |
| **Test Dice WT** | 0.7175 |
| **Test Mean Region Dice** | **0.6309** |

*Note: Ran before region Dice fix — checkpoint stores per-class mean only. Test set evaluation provides the region breakdown above.*

---

## Experiment 02 — DynUNet Baseline

**Question:** How does MONAI's nnU-Net-style DynUNet compare?

| Parameter | Value |
|-----------|-------|
| **Model** | DynUNet (MONAI) |
| **Parameters** | 22.6M |
| **Crop size** | **128 × 128 × 128** |
| **Loss** | DiceCE (dice_weight=1.0, ce_weight=1.0) |
| **Architecture** | filters=[32,64,128,256,512], 5 encoder stages, deep supervision (3 heads) |
| **SW batch size** | 2 |
| **Num workers** | 2 |

| Metric | Value |
|--------|-------|
| **Best epoch** | 260 / 300 |
| **Val Dice (per-class mean)** | 0.4052 |
| **Test Dice ET** | 0.4953 |
| **Test Dice TC** | 0.5203 |
| **Test Dice WT** | 0.6677 |
| **Test Mean Region Dice** | **0.5611** |

*Note: Ran before region Dice fix. Experienced DataLoader deadlocks requiring persistent_workers fix mid-training.*

---

## Experiment 03 — nnU-Net v2 Baseline

**Question:** Does the real nnU-Net v2 PlainConvUNet outperform MONAI's approximation?

| Parameter | Value |
|-----------|-------|
| **Model** | nnU-Net v2 PlainConvUNet (`dynamic-network-architectures`) |
| **Parameters** | 31.2M |
| **Crop size** | **128 × 128 × 128** |
| **Loss** | DiceCE (dice_weight=1.0, ce_weight=1.0) |
| **Architecture** | variant=plain, features_per_stage=[32,64,128,256,320,320], 6 encoder stages, InstanceNorm3d, LeakyReLU, deep supervision (5 heads) |
| **SW batch size** | 2 |
| **Num workers** | 2 |

| Metric | Value |
|--------|-------|
| **Best epoch** | 240 / 300 |
| **Val Dice (per-class mean)** | 0.4720 |
| **Test Dice ET** | 0.5780 |
| **Test Dice TC** | 0.6190 |
| **Test Dice WT** | 0.7263 |
| **Test Mean Region Dice** | **0.6411** |

*Note: Best overall model. The 6-stage architecture with 320-channel bottleneck and InstanceNorm outperforms DynUNet's 5-stage / 512-channel design.*

---

## Experiment 04 — SwinUNETR Baseline

**Question:** Can the Swin Transformer-based SwinUNETR compete on 8 GB VRAM?

| Parameter | Value |
|-----------|-------|
| **Model** | SwinUNETR (MONAI) |
| **Parameters** | 62.2M |
| **Crop size** | **96 × 96 × 96** ⚠️ (reduced from 128³ — OOM at 128³ on 8 GB) |
| **Loss** | DiceCE (dice_weight=1.0, ce_weight=1.0) |
| **Architecture** | feature_size=48, depths=[2,2,2,2], num_heads=[3,6,12,24] |
| **SW batch size** | 1 (reduced from 2 due to VRAM) |
| **Num workers** | 2 |

| Metric | Value |
|--------|-------|
| **Best epoch** | 150 / 300 (training incomplete) |
| **Val Dice ET** | 0.3518 |
| **Val Dice TC** | 0.3763 |
| **Val Dice WT** | 0.5543 |
| **Val Mean Region Dice** | **0.4275** |

⚠️ *SwinUNETR (62.2M params) OOMs at 128³ on 8 GB GPU. The 96³ crop significantly reduces spatial context, making this an unfair comparison. This model was designed for 32+ GB GPUs.*

---

## Experiment 05 — DiceCE Loss Study (SegResNet)

**Question:** Control experiment for loss function comparison.

| Parameter | Value |
|-----------|-------|
| **Model** | SegResNet (MONAI) |
| **Parameters** | 18.8M |
| **Crop size** | **128 × 128 × 128** |
| **Loss** | **DiceCE** (dice_weight=1.0, ce_weight=1.0) |
| **Architecture** | init_filters=32, blocks_down=[1,2,2,4], blocks_up=[1,1,1], dropout=0.2 |
| **SW batch size** | 2 |
| **Num workers** | 2 |

| Metric | Value |
|--------|-------|
| **Best epoch** | 140 / 300 (training incomplete) |
| **Val Dice ET** | 0.5355 |
| **Val Dice TC** | 0.5696 |
| **Val Dice WT** | 0.6971 |
| **Val Mean Region Dice** | **0.6008** |

---

## Experiment 06 — DiceFocal Loss Study (SegResNet)

**Question:** Does DiceFocal loss improve segmentation of small/fragmented enhancing tumor?

| Parameter | Value |
|-----------|-------|
| **Model** | SegResNet (MONAI) |
| **Parameters** | 18.8M |
| **Crop size** | **128 × 128 × 128** |
| **Loss** | **DiceFocal** (dice_weight=1.0, focal_weight=1.0, gamma=2.0) |
| **Architecture** | init_filters=32, blocks_down=[1,2,2,4], blocks_up=[1,1,1], dropout=0.2 |
| **SW batch size** | 2 |
| **Num workers** | 2 |

| Metric | Value |
|--------|-------|
| **Best epoch** | 120 / 300 (training incomplete) |
| **Val Dice ET** | 0.5168 |
| **Val Dice TC** | 0.5521 |
| **Val Dice WT** | 0.6755 |
| **Val Mean Region Dice** | **0.5815** |

---

## Comparative Summary

### Phase 1 — Model Comparison (Test Set, 129 Cases)

| Rank | Exp | Model | Params | Crop Size | Epochs | Dice ET | Dice TC | Dice WT | Mean Region |
|------|-----|-------|--------|-----------|--------|---------|---------|---------|-------------|
| 1 | 03 | **nnU-Net v2** | 31.2M | 128³ | 240 | **0.578** | **0.619** | **0.726** | **0.641** |
| 2 | 01 | SegResNet | 18.8M | 128³ | 285 | 0.572 | 0.603 | 0.718 | 0.631 |
| 3 | 02 | DynUNet | 22.6M | 128³ | 260 | 0.495 | 0.520 | 0.668 | 0.561 |
| 4 | 04 | SwinUNETR | 62.2M | 96³ ⚠️ | 150 | 0.352 | 0.376 | 0.554 | 0.427 |

### Phase 2 — Loss Function Study (Validation Set, Training Incomplete)

| Rank | Exp | Loss | Epochs | Dice ET | Dice TC | Dice WT | Mean Region |
|------|-----|------|--------|---------|---------|---------|-------------|
| 1 | 05 | **DiceCE** | 140 | **0.536** | **0.570** | **0.697** | **0.601** |
| 2 | 06 | DiceFocal | 120 | 0.517 | 0.552 | 0.676 | 0.582 |

### Key Findings

1. **nnU-Net v2 is the best model** (mean region Dice 0.641), followed closely by SegResNet (0.631)
2. **nnU-Net v2 vs DynUNet:** The real nnU-Net v2 (PlainConvUNet, 6 stages, InstanceNorm, LeakyReLU, 320-channel bottleneck) outperforms MONAI's DynUNet (5 stages, BatchNorm-style, 512-channel) by **8 points** — architecture details matter
3. **SwinUNETR is VRAM-limited:** 62.2M parameters cannot fit 128³ on 8 GB GPU. Forced to 96³ which removes 58% of spatial context, making the comparison unfair
4. **DiceCE > DiceFocal** so far — focal loss (gamma=2.0) did not improve small ET segmentation as expected. Both experiments still training
5. **WT is easiest, ET is hardest** consistently across all models (WT: 0.55–0.73, ET: 0.35–0.58)

### Gap vs Published Results

| Metric | Our Best (nnU-Net v2) | Published BraTS Winners | Gap |
|--------|----------------------|------------------------|-----|
| Dice WT | 0.726 | ~0.91 | -0.18 |
| Dice TC | 0.619 | ~0.87 | -0.25 |
| Dice ET | 0.578 | ~0.82 | -0.24 |

**Contributing factors:**
- Training duration: 240 vs 1000 epochs
- Patch size: 128³ vs 160×192×128 (limited by 8 GB GPU)
- Training data: 75% of dataset vs 100% with 5-fold cross-validation
- No post-processing (connected component analysis, threshold tuning)

## Remaining Experiments

| Exp | Description | Expected Impact |
|-----|-------------|-----------------|
| 07 | SegResNet crop 96³ | Speed vs quality tradeoff |
| 08 | SegResNet crop 160×192×128 | More context, matches original paper |
| 09 | nnU-Net v2 plain (re-run with region Dice tracking) | Cleaner metrics |
| 10 | nnU-Net v2 residual encoder (47.1M params) | Heavier encoder variant |
| 14 | DynUNet large patch 160×192×128 | Compensate for architecture with larger input |
| 15 | nnU-Net v2 patch 128×160×128 | Matches nnU-Net auto-planner |
| 16 | nnU-Net v2 patch 160×192×128 | Max patch for 8 GB GPU |
| 11 | Best model on BraTS 2023 | Cross-year validation |
| 12 | Train 2024 → test 2023 | Generalization study |
| 13 | Train 2023 → test 2024 | Generalization study |
| Native | nnU-Net v2 native pipeline | Self-configured preprocessing comparison |
