# Native nnU-Net v2 (ResEnc-M, 11 GB) — Windows Runbook (exp19)

Real-framework nnU-Net v2 replica of exp18: pools two BraTS-2024 training
releases at the **patient level**, holds out a seed-42 test set, runs 5-fold CV
with the ResEnc-M encoder planned for an **11 GB** GPU, then ensembles and
evaluates on the held-out test set.

**Fixed identifiers used throughout:**

| Thing | Value |
|---|---|
| Dataset id | `102` |
| Dataset name / tag | `BraTS2024ResEnc` / `Dataset102_BraTS2024ResEnc` |
| Plans name | `nnUNetResEncM_11GBPlans` |
| Config | `3d_fullres` |
| Trainer | `nnUNetTrainer` |
| Conda env | `gpu` (has torch+CUDA and `nnunetv2`); base at `C:\ProgramData\anaconda3` |

Pooled data dirs (both passed to the converter):

```
C:\Users\ROG STRIX\Downloads\Zain\Brats-MRI-Segmentation-For-BrainTumor\Brats2024\training_data1_v2
C:\Users\ROG STRIX\Downloads\Zain\Brats-MRI-Segmentation-For-BrainTumor\Brats2024\training_data_additional
```

**Pipeline order:** convert → **sanitize** → fingerprint → plan(11 GB) → preprocess → restore split → train → evaluate

---

## 0. Environment (every new shell)

```powershell
conda activate gpu
cd "C:\Users\ROG STRIX\Downloads\Zain\Brats-MRI-Segmentation-For-BrainTumor\brats-segmentation"
$repo = $PWD.Path
$env:nnUNet_raw          = "$repo\nnunet_data\nnUNet_raw"
$env:nnUNet_preprocessed = "$repo\nnunet_data\nnUNet_preprocessed"
$env:nnUNet_results      = "$repo\nnunet_data\nnUNet_results"
$env:CUDA_VISIBLE_DEVICES = "0"
# sanity check (expect: True <your GPU>)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Env vars are **per shell** — re-set them in every new session before running anything.

---

## 1. Convert (pool both dirs, held-out test, 5 patient-level folds)

```powershell
python nnunet_native\convert_to_nnunet.py --data_dir "C:\Users\ROG STRIX\Downloads\Zain\Brats-MRI-Segmentation-For-BrainTumor\Brats2024\training_data1_v2" "C:\Users\ROG STRIX\Downloads\Zain\Brats-MRI-Segmentation-For-BrainTumor\Brats2024\training_data_additional" --output_dir "$repo\nnunet_data" --dataset_id 102 --dataset_name BraTS2024ResEnc --mode kfold --n_folds 5 --split_seed 42
```

Images are **hard-linked** into `imagesTr` (no admin/Developer Mode needed on the
same drive); labels are re-saved as `uint8`. A `splits_final.json` with our
patient-level 5 folds is written to both `nnUNet_raw` and `nnUNet_preprocessed`.

---

## 2. Sanitize source images — CRITICAL, do NOT skip

The `training_data_additional` release ships ~44 files containing `inf` /
float-overflow voxels (`3.4e38` = float32 max, `1.8e308` = float64 max). Left in,
nnU-Net's per-image z-score normalization turns them into NaN → **`train_loss nan`
/ `val_loss nan`**, an invalid run. This scrubs the bad voxels to `0` in `imagesTr`
only; the original source files are **not** modified (the fixer drops the hard-link
and writes a fresh file). Scripts are in the Appendix.

```powershell
python nnunet_native\fix_and_verify.py "$env:nnUNet_raw\Dataset102_BraTS2024ResEnc\imagesTr"
python nnunet_native\scan_bad_images.py "$env:nnUNet_raw\Dataset102_BraTS2024ResEnc\imagesTr"   # must print 0 bad
```

Parallel workers can leave a few stragglers on the first pass — **loop fix → scan
until the scan reports `0 bad files`.**

---

## 3. Fingerprint

```powershell
nnUNetv2_extract_fingerprint -d 102 --verify_dataset_integrity
```

Runs quietly across all cases with no progress bar (several minutes) — check
`nvidia-smi`/CPU if unsure, don't assume it's stuck. `overflow encountered in cast`
warnings **after** sanitizing are cosmetic (the global intensity stats aren't used;
MRI uses per-image z-score). Done when `dataset_fingerprint.json` exists under
`nnUNet_preprocessed\Dataset102_BraTS2024ResEnc\`.

---

## 4. Plan for an 11 GB VRAM target

```powershell
nnUNetv2_plan_experiment -d 102 -pl nnUNetPlannerResEncM -gpu_memory_target 11 -overwrite_plans_name nnUNetResEncM_11GBPlans
```

`-gpu_memory_target 11` overrides the ResEnc-M preset's default budget;
`-overwrite_plans_name` writes it to a **new** plans file so the preset default is
untouched. Chosen config lands in
`nnUNet_preprocessed\Dataset102_BraTS2024ResEnc\nnUNetResEncM_11GBPlans.json`
(patch `128×192×128`, batch `2`). If training later OOMs, re-plan with
`-gpu_memory_target 9`, re-preprocess, and retrain.

---

## 5. Preprocess

```powershell
nnUNetv2_preprocess -d 102 -plans_name nnUNetResEncM_11GBPlans -c 3d_fullres -np 4
```

The slow step (resample + normalize all cases, ~12 min for ~1468 cases). Needs
plenty of free disk — see the disk note below.

---

## 6. Restore the patient-level split

Planning regenerates `splits_final.json` with nnU-Net's own random KFold — copy
ours back over it:

```powershell
Copy-Item "$env:nnUNet_raw\Dataset102_BraTS2024ResEnc\splits_final.json" "$env:nnUNet_preprocessed\Dataset102_BraTS2024ResEnc\splits_final.json" -Force
(Get-Content "$env:nnUNet_preprocessed\Dataset102_BraTS2024ResEnc\splits_final.json" | ConvertFrom-Json).Count   # expect 5
```

---

## 7. Train (fold 0; repeat for folds 1–4)

```powershell
nnUNetv2_train 102 3d_fullres 0 -p nnUNetResEncM_11GBPlans -tr nnUNetTrainer --npz
```

~162 s/epoch × 1000 epochs ≈ **45 h per fold** (≈9 days for all 5 on one GPU).
The loss is negative Dice+CE (more negative = better). **Resume** after an
interruption by appending `--c` (continues from `checkpoint_latest.pth`).

Shorter, matched schedule if time-boxed (apply to ALL folds for comparability):

```powershell
nnUNetv2_train 102 3d_fullres 0 -p nnUNetResEncM_11GBPlans -tr nnUNetTrainer_250epochs --npz
```

---

## Running long steps in the background (survive SSH disconnect)

A foreground run dies if the SSH connection drops. Use scheduled tasks — they run
independent of the login/SSH session. The wrapper `.bat` files (`win_preprocess.bat`,
`win_train_fold.bat`) set the conda env + nnU-Net env vars and log to `runs\`.

**Register with the PowerShell cmdlets** (`schtasks.exe` mangles paths with spaces →
`0x80070002`; the cmdlets pass args via COM and run `cmd.exe /c ""<bat>""`):

```powershell
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive -RunLevel Highest

$pbat = "$repo\nnunet_native\win_preprocess.bat"
Register-ScheduledTask -TaskName "nnunet_preprocess" -Principal $principal -Force `
  -Action (New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"`"$pbat`"`"")

$tbat = "$repo\nnunet_native\win_train_fold.bat"
Register-ScheduledTask -TaskName "nnunet_train_fold0" -Principal $principal -Force `
  -Action (New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"`"$tbat`"`" 0")

Start-ScheduledTask -TaskName "nnunet_preprocess"     # then, once it finishes:
Start-ScheduledTask -TaskName "nnunet_train_fold0"
```

Manage:

```powershell
schtasks /Query /TN "nnunet_train_fold0" /V /FO LIST     # status + Last Result
Stop-ScheduledTask  -TaskName "nnunet_train_fold0"       # stop a running task
schtasks /Delete    /TN "nnunet_train_fold0" /F          # remove it
```

For other folds, register with `-Argument "/c ""$tbat"" 1"` etc. and a matching `/TN`.

---

## Monitoring / logs

```powershell
$fold = "$env:nnUNet_results\Dataset102_BraTS2024ResEnc\nnUNetTrainer__nnUNetResEncM_11GBPlans__3d_fullres\fold_0"

# live per-epoch log (Ctrl+C stops only the tail, not training)
Get-Content (Get-ChildItem "$fold\training_log_*.txt" | Sort-Object LastWriteTime | Select-Object -Last 1).FullName -Wait -Tail 40

# best-Dice progression (judge convergence)
$log = (Get-ChildItem "$fold\training_log_*.txt" | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
(Select-String -Path $log -Pattern "best EMA pseudo Dice: ([0-9.]+)").Matches | ForEach-Object { $_.Groups[1].Value } | Select-Object -Last 20

# is it alive?
nvidia-smi
Get-Process python -EA SilentlyContinue | Select-Object Id, CPU, @{n='Mem(GB)';e={[math]::Round($_.WS/1GB,2)}}

# scheduled-task wrapper log (START / EXIT markers)
Get-Content "$repo\runs\train_fold0.log" -Tail 20

# checkpoints written so far
Get-ChildItem "$fold\checkpoint_*.pth" | Select-Object Name, LastWriteTime
```

The live tail of `training_log_*.txt` is the one to trust. Note: our redirected
`runs\*.log` block-buffers stdout, so it can lag — nnU-Net's own `training_log_*.txt`
is line-flushed and authoritative.

---

## Checkpoints

| File | Saved when | Use |
|---|---|---|
| `checkpoint_best.pth` | Every new best EMA pseudo Dice (the "Yayy!" lines) | Best model |
| `checkpoint_latest.pth` | Every 50 epochs | Resume point (`--c`) |
| `checkpoint_final.pth` | End of 1000 epochs | Final model |

Inference uses `final` by default — add `-chk checkpoint_best.pth` to use the best.

---

## Evaluation (after folds are trained)

```powershell
$RES = "$env:nnUNet_results\Dataset102_BraTS2024ResEnc\nnUNetTrainer__nnUNetResEncM_11GBPlans__3d_fullres"

# out-of-fold CV metric (pass the SAME pooled dirs used for conversion)
python nnunet_native\evaluate_nnunet_kfold.py --results_dir "$RES" --data_dir "<dir1>" "<dir2>" --output_dir "runs\exp19_nnunet_native_resenc_eval" --n_folds 5

# predict the held-out test set (ensemble of folds)
nnUNetv2_predict -i "$env:nnUNet_raw\Dataset102_BraTS2024ResEnc\imagesTs" -o "$RES\test_predictions" -d 102 -c 3d_fullres -p nnUNetResEncM_11GBPlans -tr nnUNetTrainer -f 0 1 2 3 4 -chk checkpoint_best.pth

# score the test predictions with the shared metrics
python nnunet_native\evaluate_nnunet.py --pred_dir "$RES\test_predictions" --data_dir "<dir1>" "<dir2>" --output_dir "runs\exp19_nnunet_native_resenc_test_eval"
```

Replace `<dir1> <dir2>` with the two `training_data*` paths from section 0. Passing
the identical pooled dirs is required so the seed-42 test split reconstructs exactly.

---

## Gotchas we hit (and the fixes)

- **Ctrl+C does nothing over SSH + conpty.** Stop from a 2nd session:
  `taskkill /F /T /PID <pid>` or `Get-Process python | Stop-Process -Force`.
- **Log "frozen" at one line ≠ hung.** Redirected stdout is block-buffered, and
  startup is quiet (CUDA init, unpacking dataset). Watch `training_log_*.txt` or
  `nvidia-smi` instead.
- **`schtasks /TR` with a spaced path → `0x80070002` (file not found).** Use
  `Register-ScheduledTask` + `cmd.exe /c ""<bat>""` (doubled quotes), as above.
- **`train_loss nan`** → corrupt source voxels. Run the sanitizer (step 2) BEFORE
  fingerprint/preprocess. This is baked into the pipeline order here.
- **Plateau around epoch 450 is deceptive.** The poly-LR tail (last ~30%, LR decaying
  to 0) still adds ~1–3% Dice; 1000 epochs is the comparable standard.
- **Disk:** preprocessing writes the large `.npy` set and training unpacks it. Keep
  **≥150 GB free** on C: before step 5.

---

## Quick re-run cheat sheet

```
0  conda activate gpu; cd repo; set the four $env: vars
1  convert:      python nnunet_native\convert_to_nnunet.py --data_dir <d1> <d2> --output_dir <repo>\nnunet_data --dataset_id 102 --dataset_name BraTS2024ResEnc --mode kfold --n_folds 5 --split_seed 42
2  sanitize:     python nnunet_native\fix_and_verify.py <imagesTr>  ;  python nnunet_native\scan_bad_images.py <imagesTr>   (loop until 0 bad)
3  fingerprint:  nnUNetv2_extract_fingerprint -d 102 --verify_dataset_integrity
4  plan (11GB):  nnUNetv2_plan_experiment -d 102 -pl nnUNetPlannerResEncM -gpu_memory_target 11 -overwrite_plans_name nnUNetResEncM_11GBPlans
5  preprocess:   nnUNetv2_preprocess -d 102 -plans_name nnUNetResEncM_11GBPlans -c 3d_fullres -np 4
6  restore split: copy splits_final.json  raw -> preprocessed
7  train:        nnUNetv2_train 102 3d_fullres <fold> -p nnUNetResEncM_11GBPlans -tr nnUNetTrainer --npz     (folds 0..4; --c to resume)
8  evaluate:     evaluate_nnunet_kfold.py (CV)  ;  nnUNetv2_predict imagesTs (-f 0 1 2 3 4)  ;  evaluate_nnunet.py (test)
```

---

## Appendix — sanitizer scripts

Create these once under `nnunet_native\`.

**`scan_bad_images.py`** — detect non-finite / overflow voxels:

```python
import sys, glob, os
import numpy as np, SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed


def check(f):
    try:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float64)
    except Exception as e:
        return (f, "READ_ERROR", str(e))
    finite = np.isfinite(arr)
    n_bad = int((~finite).sum())
    amax = float(np.abs(arr[finite]).max()) if finite.any() else float("inf")
    if n_bad > 0 or amax > 1e6:
        return (f, n_bad, amax)
    return None


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.nii.gz")))
    print(f"Scanning {len(files)} files...")
    bad = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for fut in as_completed([ex.submit(check, f) for f in files]):
            r = fut.result()
            if r:
                bad.append(r)
                print("BAD:", os.path.basename(r[0]), "nonfinite=", r[1], "absmax=", r[2])
    cases = sorted(set(os.path.basename(f).rsplit("_", 1)[0] for f, _, _ in bad))
    print(f"\n{len(bad)} bad files across {len(cases)} cases:")
    for c in cases:
        print(" ", c)
```

**`fix_and_verify.py`** — repair (bad voxels → 0) and re-read to confirm:

```python
import sys, glob, os
import numpy as np, SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor

THRESH = 1e6


def bad_mask(a):
    return ~np.isfinite(a) | (np.abs(a) > THRESH)


def fix_one(f):
    a = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float64)
    m = bad_mask(a)
    n = int(m.sum())
    if n == 0:
        return (os.path.basename(f), 0, True)
    a[m] = 0.0
    out = sitk.GetImageFromArray(a.astype(np.float32))
    out.CopyInformation(sitk.ReadImage(f))   # preserve spacing/origin/direction
    os.remove(f)                             # drop hard-link; source untouched
    sitk.WriteImage(out, f)                  # write fresh, sanitized file
    chk = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float64)
    return (os.path.basename(f), n, not bool(bad_mask(chk).any()))


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*.nii.gz")))
    print(f"Processing {len(files)} files...")
    fixed, failed = 0, []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for name, n, ok in ex.map(fix_one, files):
            if n:
                fixed += 1
                print(("FIXED     " if ok else "STILL BAD ") + name + f"  voxels={n}")
                if not ok:
                    failed.append(name)
    print(f"\nSanitized {fixed} files. Verification failures: {len(failed)}")
    for x in failed:
        print("  FAILED:", x)
```
