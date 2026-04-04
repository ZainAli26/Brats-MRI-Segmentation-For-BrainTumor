# BraTS MRI Segmentation

Multi-model brain tumor segmentation pipeline for BraTS 2023/2024 datasets.
Compares four architectures under identical preprocessing, evaluates per-class
and per-region metrics, and provides visual debugging for failure cases.

## Project Structure

```
brats-segmentation/
├── train.py                          # Training entrypoint
├── evaluate.py                       # Evaluation + visualization entrypoint
├── analyze_failures.py               # Standalone failure analysis & model comparison
├── requirements.txt
├── configs/
│   ├── config.yaml                   # BraTS 2024 (default)
│   └── config_brats2023.yaml         # BraTS 2023
├── src/
│   ├── data/
│   │   ├── splits.py                 # Patient-level train/val/test splitting
│   │   ├── preprocessing.py          # Shared MONAI transforms (all models)
│   │   └── dataset.py                # CacheDataset + DataLoaders
│   ├── models/
│   │   └── factory.py                # Model factory (4 architectures)
│   ├── training/
│   │   ├── trainer.py                # Training loop, validation, checkpointing
│   │   └── losses.py                 # DiceCE, DiceFocal, deep supervision wrapper
│   ├── evaluation/
│   │   ├── metrics.py                # Per-case Dice, HD95, region metrics
│   │   ├── failure_analysis.py       # Identifies low-Dice, small tumor, missed ET
│   │   └── visualization.py          # Overlays, box plots, scatter, model comparison
│   └── utils/
│       └── experiment.py             # Run naming, config saving, TensorBoard
├── nnunet_native/                    # Native nnU-Net v2 pipeline (separate from custom loop)
│   ├── convert_to_nnunet.py          # BraTS → nnU-Net format converter
│   ├── run_nnunet.sh                 # Full pipeline runner (plan → train → predict)
│   └── evaluate_nnunet.py            # Bridge: nnU-Net predictions → our metrics
└── runs/                             # Auto-created: checkpoints, logs, results
```

## Setup

```bash
cd brats-segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Layout

Place datasets under the repo root:

```
Brats-MRI-Segmentation-For-BrainTumor/
├── Brats2024/
│   ├── training_data1_v2/    # Cases with seg masks
│   └── validation_data/      # Cases without seg masks
├── Brats2023/
│   ├── training_data/
│   └── validation_data/
└── brats-segmentation/       # This project
```

Each case directory contains NIfTI files:
`{case_id}-t1c.nii.gz`, `{case_id}-t1n.nii.gz`, `{case_id}-t2f.nii.gz`, `{case_id}-t2w.nii.gz`, `{case_id}-seg.nii.gz`

## Models

| Model | CLI Name | Architecture | Params | Source |
|-------|----------|-------------|--------|--------|
| nnU-Net v2 | `nnunet_v2` | PlainConvUNet / ResidualEncoderUNet | 31.2M | `dynamic-network-architectures` (nnunetv2) |
| DynUNet | `dynunet` | MONAI's nnU-Net-style UNet | 22.6M | `monai` |
| SwinUNETR | `swin_unetr` | Swin Transformer + UNet decoder | 62.2M | `monai` |
| SegResNet | `segresnet` | ResNet-based encoder-decoder | 18.8M | `monai` |

## Training

### Quick Start

```bash
# Train with default config (SegResNet on BraTS 2024)
python train.py

# Train a specific model
python train.py --model nnunet_v2
python train.py --model dynunet
python train.py --model swin_unetr
python train.py --model segresnet
```

### Switching Datasets

```bash
# BraTS 2024 (default)
python train.py --config configs/config.yaml --model segresnet

# BraTS 2023
python train.py --config configs/config_brats2023.yaml --model segresnet
```

### CLI Overrides

Any config value can be overridden from the command line:

```bash
python train.py --model nnunet_v2 --epochs 500 --batch_size 1 --lr 0.0002
python train.py --model swin_unetr --epochs 200 --batch_size 1  # larger model, smaller batch
```

### Resume Training

```bash
python train.py --model segresnet --resume runs/segresnet_20260329_120000/best_model.pth
```

### Run All Models (Full Experiment)

```bash
for model in nnunet_v2 dynunet swin_unetr segresnet; do
    python train.py --model $model 2>&1 | tee runs/${model}_training.log
done
```

## Configuration Reference

All settings live in `configs/config.yaml`. Key parameters:

### Data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.train_dir` | `../Brats2024/training_data1_v2` | Path to training cases |
| `data.split_ratios` | `[0.75, 0.15, 0.10]` | Train/val/test split (patient-level) |
| `data.split_seed` | `42` | Random seed for reproducible splits |

### Preprocessing (Identical Across All Models)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `preprocessing.spatial_size` | `[128, 128, 128]` | Crop/pad target size |
| `preprocessing.norm_method` | `zscore_nonzero` | Z-score on nonzero voxels per channel |
| `augmentation.random_flip_prob` | `0.5` | Random flip probability (per axis) |
| `augmentation.random_rotate_prob` | `0.3` | Random 90-degree rotation probability |
| `augmentation.random_intensity_shift` | `0.1` | Additive intensity offset range |
| `augmentation.random_intensity_scale` | `0.1` | Multiplicative intensity scale range |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | `300` | Maximum epochs |
| `training.batch_size` | `2` | Batch size (reduce to 1 for SwinUNETR on 12GB GPU) |
| `training.learning_rate` | `0.0001` | Initial learning rate |
| `training.weight_decay` | `0.00001` | AdamW weight decay |
| `training.optimizer` | `adamw` | Optimizer (`adamw` or `adam`) |
| `training.scheduler` | `cosine_warm_restarts` | LR scheduler |
| `training.amp` | `true` | Mixed precision (FP16) training |
| `training.grad_accum_steps` | `1` | Gradient accumulation (increase to simulate larger batch) |
| `training.val_interval` | `5` | Validate every N epochs |
| `training.early_stopping_patience` | `30` | Stop after N val checks without improvement |
| `training.loss` | `dice_ce` | Loss function (`dice_ce` or `dice_focal`) |
| `training.sw_batch_size` | `4` | Sliding window batch size for validation inference |
| `training.sw_overlap` | `0.5` | Sliding window overlap ratio |

### Model-Specific

**nnU-Net v2** (`--model nnunet_v2`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nnunet_v2.variant` | `plain` | `plain` (PlainConvUNet) or `residual` (ResidualEncoderUNet) |
| `nnunet_v2.features_per_stage` | `[32,64,128,256,320,320]` | Feature channels per encoder stage |
| `nnunet_v2.deep_supervision` | `true` | Multi-scale loss on decoder outputs |
| `nnunet_v2.n_blocks_encoder` | `2` | Conv blocks per encoder stage |
| `nnunet_v2.n_blocks_decoder` | `2` | Conv blocks per decoder stage |

**DynUNet** (`--model dynunet`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dynunet.filters` | `[32,64,128,256,512]` | Feature channels per stage |
| `dynunet.deep_supervision` | `true` | Deep supervision |
| `dynunet.deep_supervision_heads` | `3` | Number of deep supervision outputs |

**SwinUNETR** (`--model swin_unetr`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `swin_unetr.feature_size` | `48` | Swin Transformer feature dimension |
| `swin_unetr.depths` | `[2,2,2,2]` | Transformer blocks per stage |
| `swin_unetr.num_heads` | `[3,6,12,24]` | Attention heads per stage |
| `swin_unetr.drop_rate` | `0.0` | Dropout rate |

**SegResNet** (`--model segresnet`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segresnet.init_filters` | `32` | Initial convolution filters |
| `segresnet.blocks_down` | `[1,2,2,4]` | ResBlocks per encoder stage |
| `segresnet.blocks_up` | `[1,1,1]` | ResBlocks per decoder stage |
| `segresnet.dropout_prob` | `0.2` | Dropout probability |

## Evaluation

### Evaluate a Trained Model

```bash
# Evaluate on test split (default)
python evaluate.py --run_dir runs/segresnet_20260329_120000

# Evaluate on validation split
python evaluate.py --run_dir runs/segresnet_20260329_120000 --split val

# Include visual overlays for failure cases
python evaluate.py --run_dir runs/segresnet_20260329_120000 --visualize_failures
```

### Compare Multiple Models

```bash
python evaluate.py \
    --run_dir runs/segresnet_20260329_120000 \
    --compare runs/nnunet_v2_20260329_130000 runs/swin_unetr_20260329_140000
```

This generates:
- `case_metrics.csv` — per-case Dice, HD95 for all classes and regions
- `metrics_distributions.png` — box plots and histograms
- `model_comparison.png` — side-by-side Dice box plots across models
- `training_curves.png` — loss and validation Dice over epochs

## Failure Analysis

### Analyze a Single Run

```bash
python analyze_failures.py --run_dir runs/segresnet_20260329_120000
```

Detects and reports:
- **Low overall Dice** — cases below threshold (default 0.5)
- **ET failures** — poor enhancing tumor segmentation
- **Small tumor failures** — small ET regions (<500 voxels) with low Dice
- **Over-segmentation** — predicted volume >2x true volume
- **Missed ET** — ET present in ground truth but absent in prediction

### Visualize Specific Problem Cases

```bash
python analyze_failures.py \
    --run_dir runs/segresnet_20260329_120000 \
    --cases BraTS-GLI-00463-100 BraTS-GLI-00528-101
```

Generates per-case 3-view (axial/coronal/sagittal) comparison images showing
ground truth vs prediction with tumor overlay and Dice scores.

### Compare Failure Patterns Across Models

```bash
python analyze_failures.py \
    --run_dirs runs/nnunet_v2_* runs/dynunet_* runs/segresnet_* runs/swin_unetr_* \
    --compare
```

## Native nnU-Net v2 Pipeline

Run nnU-Net v2 with its own preprocessing and training for comparison.

### Full Pipeline (One Command)

```bash
bash nnunet_native/run_nnunet.sh --data_dir ../Brats2024/training_data1_v2 --gpu 0
```

### Step by Step

```bash
# 1. Convert BraTS data to nnU-Net format (symlinks + label remapping)
python nnunet_native/convert_to_nnunet.py --data_dir ../Brats2024/training_data1_v2

# 2. Set environment variables
export nnUNet_raw=./nnunet_data/nnUNet_raw
export nnUNet_preprocessed=./nnunet_data/nnUNet_preprocessed
export nnUNet_results=./nnunet_data/nnUNet_results

# 3. Plan and preprocess
nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres --verify_dataset_integrity

# 4. Train (fold 0 uses our patient-level split)
nnUNetv2_train 101 3d_fullres 0

# 5. Predict on test set
nnUNetv2_predict \
    -i nnunet_data/nnUNet_raw/Dataset101_BraTS2024/imagesTs \
    -o predictions -d 101 -c 3d_fullres -f 0

# 6. Evaluate using our shared metrics
python nnunet_native/evaluate_nnunet.py --pred_dir predictions
```

### Compare Native vs Custom-Loop Results

```bash
python analyze_failures.py \
    --run_dirs runs/nnunet_native_eval runs/nnunet_v2_20260329_* \
    --compare
```

## BraTS Data Format

### Modalities

| Suffix | Full Name |
|--------|-----------|
| `t1n` | T1-weighted native |
| `t1c` | T1-weighted contrast-enhanced |
| `t2w` | T2-weighted |
| `t2f` | T2-weighted FLAIR |

### Segmentation Labels

| Original Label | Remapped | Structure | Overlay Color |
|---------------|----------|-----------|---------------|
| 0 | 0 | Background | — |
| 1 | 1 | NCR (Necrotic Core) | Red |
| 2 | 2 | ED (Peritumoral Edema) | Green |
| 4 | 3 | ET (Enhancing Tumor) | Blue |

### Evaluation Regions

| Region | Labels | Description |
|--------|--------|-------------|
| ET | 4 (→3) | Enhancing Tumor |
| TC | 1 + 4 (→1,3) | Tumor Core |
| WT | 1 + 2 + 4 (→1,2,3) | Whole Tumor |

## Patient-Level Splitting

BraTS case IDs follow the pattern `BraTS-GLI-XXXXX-YYY` where `XXXXX` is the
patient ID and `YYY` is the longitudinal timepoint. The pipeline groups all
timepoints of the same patient and assigns them to the same split to prevent
data leakage. This is enforced in both the custom training loop and the native
nnU-Net pipeline (via `splits_final.json`).

## Experiment Tracking

Each training run creates a timestamped directory under `runs/`:

```
runs/segresnet_20260329_120000/
├── config.yaml              # Frozen config for this run
├── best_model.pth           # Best checkpoint (by val Dice)
├── summary.json             # Final metrics summary
├── training.log             # Scalar log (CSV format)
├── logs/                    # TensorBoard events
├── visualizations/          # Generated plots
└── eval_test/               # Evaluation results
    ├── case_metrics.csv
    ├── metrics_distributions.png
    ├── failures_*.csv
    └── case_visualizations/
```

View TensorBoard logs:

```bash
tensorboard --logdir runs/
```

## GPU Memory Guide

| Model | batch_size=2 | batch_size=1 | With grad_accum=2 |
|-------|-------------|-------------|-------------------|
| SegResNet (18.8M) | ~8 GB | ~5 GB | ~5 GB (effective batch 2) |
| DynUNet (22.6M) | ~10 GB | ~6 GB | ~6 GB |
| nnU-Net v2 (31.2M) | ~12 GB | ~7 GB | ~7 GB |
| SwinUNETR (62.2M) | ~18 GB | ~10 GB | ~10 GB |

If running on a 12 GB GPU, use `--batch_size 1` for SwinUNETR and nnU-Net v2.
Use `grad_accum_steps: 2` in config to compensate for smaller batch size.

## Docker

### Build

```bash
cd brats-segmentation
docker build -t brats-seg .
```

### Install docker-compose (if not already installed)

```bash
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
    -o /usr/local/bin/docker-compose && sudo chmod +x /usr/local/bin/docker-compose
```

### Docker Compose (Recommended)

The simplest way to run experiments. All GPU, volume, and path configuration
is handled automatically by `docker-compose.yml`.

```bash
# Build the image
docker-compose build

# --- Run experiments one at a time ---
docker-compose run --rm exp01          # Phase 1: SegResNet baseline
docker-compose run --rm exp02          # Phase 1: DynUNet baseline
docker-compose run --rm exp03          # Phase 1: nnU-Net v2 baseline
docker-compose run --rm exp04          # Phase 1: SwinUNETR baseline
docker-compose run --rm exp05          # Phase 2: DiceCE loss
docker-compose run --rm exp06          # Phase 2: DiceFocal loss
docker-compose run --rm exp07          # Phase 3: Crop 96^3
docker-compose run --rm exp08          # Phase 3: Crop 160^3
docker-compose run --rm exp09          # Phase 4: nnU-Net v2 plain
docker-compose run --rm exp10          # Phase 4: nnU-Net v2 residual
docker-compose run --rm native         # Phase 5: Native nnU-Net v2
docker-compose run --rm exp11          # Phase 6: BraTS 2023
docker-compose run --rm exp12          # Phase 7: Train 2024 → test 2023
docker-compose run --rm exp13          # Phase 7: Train 2023 → test 2024

# --- Evaluate & compare ---
docker-compose run --rm eval                                          # Evaluate latest run
docker-compose run --rm -e RUN_NAME=segresnet_20260329_120000 eval-run  # Evaluate specific run
docker-compose run --rm compare                                       # Compare all runs

# --- TensorBoard ---
docker-compose up tensorboard          # Open http://localhost:6006

# --- Interactive shell ---
docker-compose run --rm shell
```

### Data Mount Convention (for docker run)

If you prefer plain `docker run` over compose, set these once per session:

```bash
export BRATS_ROOT=/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor
export BRATS_SEG=$BRATS_ROOT/brats-segmentation
```

### Run a Single Experiment

```bash
# Phase 1 — Baseline models
docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_ROOT/Brats2023:/data/Brats2023:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp01_segresnet_baseline.yaml \
        --data_dir /data/Brats2024/training_data1_v2
```

Replace `exp01` with any experiment config (exp01–exp13). For BraTS 2023
experiments (exp11, exp13), use `--data_dir /data/Brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`.

### All Experiment Commands

```bash
# --- Phase 1: Baselines ---
docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp01_segresnet_baseline.yaml \
        --data_dir /data/Brats2024/training_data1_v2

docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp02_dynunet_baseline.yaml \
        --data_dir /data/Brats2024/training_data1_v2

docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp03_nnunet_v2_baseline.yaml \
        --data_dir /data/Brats2024/training_data1_v2

docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp04_swin_unetr_baseline.yaml \
        --data_dir /data/Brats2024/training_data1_v2

# --- Phase 2: Loss functions ---
docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp05_dice_ce.yaml \
        --data_dir /data/Brats2024/training_data1_v2

docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp06_dice_focal.yaml \
        --data_dir /data/Brats2024/training_data1_v2

# --- Phase 3: Resolution ---
docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp07_crop96.yaml \
        --data_dir /data/Brats2024/training_data1_v2

docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp08_crop160.yaml \
        --data_dir /data/Brats2024/training_data1_v2

# --- Phase 4: nnU-Net v2 variants ---
docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp09_nnunet_v2_plain.yaml \
        --data_dir /data/Brats2024/training_data1_v2

docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp10_nnunet_v2_residual.yaml \
        --data_dir /data/Brats2024/training_data1_v2

# --- Phase 5: Native nnU-Net v2 ---
docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    -v $BRATS_SEG/nnunet_data:/app/nnunet_data \
    -e nnUNet_raw=/app/nnunet_data/nnUNet_raw \
    -e nnUNet_preprocessed=/app/nnunet_data/nnUNet_preprocessed \
    -e nnUNet_results=/app/nnunet_data/nnUNet_results \
    brats-seg bash nnunet_native/run_nnunet.sh \
        --data_dir /data/Brats2024/training_data1_v2 --gpu 0

# --- Phase 6: BraTS 2023 ---
docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2023:/data/Brats2023:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp11_brats2023_best.yaml \
        --data_dir /data/Brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData

# --- Phase 7: Cross-dataset ---
docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp12_train2024_test2023.yaml \
        --data_dir /data/Brats2024/training_data1_v2

docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2023:/data/Brats2023:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python train.py \
        --config experiments/exp13_train2023_test2024.yaml \
        --data_dir /data/Brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
```

### Evaluate Inside Docker

```bash
# Evaluate the latest run
docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python evaluate.py \
        --run_dir runs/<run_name> --visualize_failures

# Compare all runs
docker run --rm --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg python analyze_failures.py \
        --run_dirs runs/*/ --compare
```

### TensorBoard

```bash
docker run --rm -p 6006:6006 \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg tensorboard --logdir runs --host 0.0.0.0 --port 6006
# Open http://localhost:6006
```

### Interactive Shell

```bash
docker run --rm -it --gpus all --shm-size=8g \
    -v $BRATS_ROOT/Brats2024:/data/Brats2024:ro \
    -v $BRATS_ROOT/Brats2023:/data/Brats2023:ro \
    -v $BRATS_SEG/runs:/app/runs \
    brats-seg bash
```
