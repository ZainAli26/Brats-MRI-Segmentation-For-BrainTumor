# BraTS Segmentation — Progress Report

_Last updated: 2026-06-17_

## 1. Where We Are

We have completed the **baseline model comparison** and the **loss-function study**,
and we have a **first full-quality 5-fold result** from the residual-encoder nnU-Net v2
(exp18, fold 0). Alongside the experiments, the **evaluation, post-processing, and
visualization infrastructure** is now built and validated.

**Current best model:** nnU-Net v2 **ResidualEncoderUNet**, 160×192×128 patch,
5-fold CV — **fold 0 = 0.664 mean region Dice** (ET 0.605 / TC 0.641 / WT 0.746).
This is the strongest result to date, ahead of the previous best (exp03 plain
nnU-Net v2 at 0.641 on the test set).

---

## 2. Experiments Completed & Evaluated

### Phase 1 — Model comparison (test set, 129 cases)

| Rank | Exp | Model | Params | Patch | Epochs | ET | TC | WT | **Mean** |
|------|-----|-------|--------|-------|--------|-----|-----|-----|---------|
| 1 | 03 | nnU-Net v2 (plain) | 31.2M | 128³ | 240 | 0.578 | 0.619 | 0.726 | **0.641** |
| 2 | 01 | SegResNet | 18.8M | 128³ | 285 | 0.572 | 0.603 | 0.718 | **0.631** |
| 3 | 02 | DynUNet | 22.6M | 128³ | 260 | 0.495 | 0.520 | 0.668 | **0.561** |
| 4 | 04 | SwinUNETR ⚠️ | 62.2M | 96³ | 150 | 0.352 | 0.376 | 0.554 | **0.427** |

⚠️ SwinUNETR was forced to 96³ (OOM at 128³ on 8 GB) — an unfair, VRAM-limited comparison.

### Phase 2 — Loss study (validation set, training incomplete)

| Exp | Loss | Epochs | ET | TC | WT | **Mean** |
|-----|------|--------|-----|-----|-----|---------|
| 05 | DiceCE | 140 | 0.536 | 0.570 | 0.697 | **0.601** |
| 06 | DiceFocal | 120 | 0.517 | 0.552 | 0.676 | **0.582** |

DiceCE ≥ DiceFocal so far; both runs stopped early (120–140 epochs).

### Phase 4 — Best architecture, full pipeline (exp18, validation fold 0)

| Exp | Model | Params | Patch | CV | Best epoch | ET | TC | WT | **Mean** |
|-----|-------|--------|-------|-----|-----------|-----|-----|-----|---------|
| 18  | nnU-Net v2 **residual** | 47.1M | 160×192×128 | 5-fold (fold 0) | 280 | 0.605 | 0.641 | 0.746 | **0.664** |

- **+0.023 mean Dice** over the previous best (exp03), driven by the residual encoder,
  the larger patch (160×192×128 vs 128³), and full-data k-fold splitting.
- Folds 1–4 are **not yet trained** — so no ensemble result exists yet.

---

## 3. Infrastructure Delivered

| Component | Status | Notes |
|-----------|--------|-------|
| `train_kfold.py` 5-fold training | ✅ | `--smoke_test`, `--max_epochs`, `--fold`, resume support |
| GPU training optimizations | ✅ | GPU-side loss/Dice accumulation, `non_blocking` transfers, `cudnn.benchmark`, vectorized region Dice |
| `evaluate_kfold.py` | ✅ | Single-fold + N-fold **ensemble**, **TTA**, post-proc toggle, per-case CSV + summary JSON |
| `src/evaluation/postprocessing.py` | ✅ | ET min-voxel threshold, connected-component filtering, hole filling |
| `visualize_segmentations.py` | ✅ | 2D overlay PNGs of GT vs prediction |
| `export_3d_html.py` | ✅ | Interactive 3D HTML renders (plotly) |
| Windows / conda run docs | ✅ | `WINDOWS_COMMANDS.md` |

---

## 4. Key Findings So Far

1. **Architecture matters most.** Real nnU-Net v2 (InstanceNorm, LeakyReLU, 6 stages,
   320-ch bottleneck) beat MONAI's DynUNet by ~8 Dice points; the **residual encoder**
   then added another ~2 points over the plain variant.
2. **ET is the hard region** across every model (0.35–0.61); WT is consistently easiest
   (0.55–0.75). Small/fragmented enhancing tumor is the main accuracy gap.
3. **Post-processing currently does nothing measurable** — on fold 0, post-processed
   mean Dice (0.6639) ≈ raw (0.6641). The current params (ET min 250 voxels, min
   component 50, fill holes) need tuning or they are no-ops on this model's output.
4. **Loss:** DiceFocal (γ=2.0) did **not** help small-ET segmentation as hoped.

### Gap vs published BraTS winners

| Region | Our best (exp18 fold0) | Published ~ | Gap |
|--------|------------------------|-------------|-----|
| WT | 0.746 | 0.91 | −0.16 |
| TC | 0.641 | 0.87 | −0.23 |
| ET | 0.605 | 0.82 | −0.21 |

Contributing factors: training length (280 vs ~1000 epochs), single fold (no ensemble),
no TTA, and un-tuned post-processing.

---

## 5. Next Steps (Remaining Work)

Ordered by expected impact on the leaderboard metric.

### Immediate (close out the current best model)
1. **Train exp18 folds 1–4.** Only fold 0 is done. This is the single biggest item —
   a 5-fold ensemble typically adds several Dice points on its own.
2. **Ensemble + TTA evaluation** once folds complete:
   `evaluate_kfold.py --fold_dirs <fold0..4> --tta`. Compare ensemble vs single-fold.
3. **Investigate the aborted May-22 fold-0 run** (empty checkpoints) — confirm it's not
   masking a config/OOM regression before launching folds 1–4.

### Accuracy improvements
4. **Tune post-processing.** It's currently a no-op. Sweep ET min-voxel threshold
   (e.g. 0 / 100 / 250 / 500) — the standard BraTS trick of zeroing tiny ET
   components, which directly targets our weakest region.
5. **Longer training / re-run incomplete experiments.** exp05/06 stopped at 120–140
   epochs; exp18 at 280. Confirm convergence headroom before drawing loss conclusions.

### Validation & generalization
6. **exp19 — native nnU-Net v2 comparison.** Config is ready; runs the same residual
   5-fold setup through the real nnU-Net framework (its own planner/preprocessing) to
   quantify how much our custom loop leaves on the table.
7. **Cross-dataset studies (exp11–13):** best model on BraTS 2023, and the
   2024→2023 / 2023→2024 generalization runs.

### Reporting
8. **Update `RESULTS.md`** to fold in exp18 and the post-processing/TTA findings
   (it currently documents only the first 6 experiments).

---

## 6. One-Line Status per Planned Experiment

| Exp | Description | Status |
|-----|-------------|--------|
| 01–04 | Phase 1 model baselines | ✅ done & evaluated |
| 05–06 | Loss study (DiceCE / DiceFocal) | ⚠️ trained partially, evaluated on val |
| 07–08 | SegResNet crop 96³ / 160×192×128 | ⬜ config ready, not run |
| 09–10 | nnU-Net v2 plain / residual (single split) | ⬜ config ready, not run |
| 14–16 | Large/max-patch variants | ⬜ config ready, not run |
| 17 | nnU-Net v2 plain 5-fold | ⬜ config ready, not run |
| **18** | **nnU-Net v2 residual 5-fold (current best)** | 🟡 **fold 0 done; folds 1–4 pending** |
| 19 | Native nnU-Net v2 residual 5-fold | ⬜ config ready, not run |
| 11–13 | BraTS 2023 + cross-dataset generalization | ⬜ config ready, not run |
