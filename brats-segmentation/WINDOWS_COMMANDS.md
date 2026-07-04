# Windows Commands (conda, no Docker)

Copy-paste-ready commands for running training and evaluation on Windows with a
conda environment. Run all commands from the `brats-segmentation\` directory
unless noted otherwise.

> **Adjust before running:** replace `C:\data\Brats2024\training_data` with your
> real dataset path, and replace the `runs\..._fold0` run-directory names with the
> actual timestamped folders created under `runs\` during training.

---

## 1. One-time environment setup

```bat
conda create -n brats python=3.12 -y
conda activate brats

:: Install CUDA-enabled PyTorch FIRST so the CPU-only wheel isn't pulled in
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

:: Then the rest of the dependencies
pip install -r requirements.txt
```

Verify the GPU is detected (expect `True` + your GPU name):

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Activate the env in every new terminal before running anything below:

```bat
conda activate brats
```

---

## 2. Training

### Quick smoke test (validate setup — 3 epochs, single fold)

```bat
python train_kfold.py --config experiments\exp18_nnunet_v2_residual_5fold.yaml --data_dir C:\data\Brats2024\training_data --smoke_test
```

### Single experiment (single model, full run)

```bat
python train.py --config experiments\exp18_nnunet_v2_residual_5fold.yaml
```

### Full 5-fold cross-validation training

```bat
python train_kfold.py --config experiments\exp18_nnunet_v2_residual_5fold.yaml --data_dir C:\data\Brats2024\training_data
```

### Train a single fold only (e.g. fold 0)

```bat
python train_kfold.py --config experiments\exp18_nnunet_v2_residual_5fold.yaml --data_dir C:\data\Brats2024\training_data --fold 0
```

### Cap epochs (for a quicker local run)

```bat
python train_kfold.py --config experiments\exp18_nnunet_v2_residual_5fold.yaml --data_dir C:\data\Brats2024\training_data --max_epochs 50
```

---

## 3. Evaluation

### Evaluate a single fold checkpoint

```bat
python evaluate_kfold.py --fold_dirs runs\nnunet_v2_20260507_021354_fold0 --config experiments\exp18_nnunet_v2_residual_5fold.yaml
```

### Ensemble all 5 folds

```bat
python evaluate_kfold.py --fold_dirs runs\fold0 runs\fold1 runs\fold2 runs\fold3 runs\fold4 --config experiments\exp18_nnunet_v2_residual_5fold.yaml
```

### Evaluate with test-time augmentation (TTA)

```bat
python evaluate_kfold.py --fold_dirs runs\nnunet_v2_20260507_021354_fold0 --config experiments\exp18_nnunet_v2_residual_5fold.yaml --tta
```

### Evaluate raw predictions (skip post-processing)

```bat
python evaluate_kfold.py --fold_dirs runs\nnunet_v2_20260507_021354_fold0 --config experiments\exp18_nnunet_v2_residual_5fold.yaml --no_postproc
```

### Single-model evaluation (non-kfold runs)

```bat
python evaluate.py --run_dir runs\segresnet_20240101_120000 --visualize_failures
```

---

## 4. Visualization

### 2D overlay PNGs (defaults to .\visualizations\)

```bat
python visualize_segmentations.py --fold_dirs runs\nnunet_v2_20260507_021354_fold0 --config experiments\exp18_nnunet_v2_residual_5fold.yaml --max_cases 20
```

### Interactive 3D HTML export (defaults to .\visualizations_3d\)

```bat
python export_3d_html.py --fold_dirs runs\nnunet_v2_20260507_021354_fold0 --config experiments\exp18_nnunet_v2_residual_5fold.yaml --max_cases 10
```

### Failure analysis

```bat
python analyze_failures.py --run_dir runs\segresnet_20240101_120000
```

---

## Notes

- **`--data_dir`** overrides the `data.train_dir` set in the experiment YAML — use it
  to point at your Windows dataset path instead of editing the config.
- If DataLoader workers cause problems, lower `num_workers` (default `6` in the
  exp18 config; eval/visualize scripts accept `--num_workers 2`). Windows spawns
  workers (no `fork`), so high worker counts have more overhead than on Linux.
- **TensorBoard** to watch training: `tensorboard --logdir runs`
- Long lines: Windows `cmd` does not support `\` line continuation. Keep each
  command on a single line (as written above), or use `^` at line ends in `cmd`
  / backtick `` ` `` in PowerShell if you want to split them.
