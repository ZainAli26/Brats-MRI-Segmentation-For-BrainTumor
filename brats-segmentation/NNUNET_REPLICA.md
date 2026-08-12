# nnU-Net Replica — a custom loop that reproduces the native run

`src/nnunet_replica/` reimplements nnU-Net v2's **entire training procedure** in code we
own, driven by a real `plans.json`. The target being reproduced is

```
plans/nnUNetResEncM_11GBPlans.json   (nnUNetPlannerResEncM, 11 GB VRAM, Dataset102_BraTS2024ResEnc)
  3d_fullres:  patch [128, 192, 128]   batch 2   spacing 1 mm   batch_dice False
  ResidualEncoderUNet, 6 stages, features [32, 64, 128, 256, 320, 320],
  encoder blocks [1, 3, 4, 6, 6, 6], decoder convs [1, 1, 1, 1, 1]  ->  102,354,777 params
```

Once exp20 matches the native numbers, every later experiment is a **single-variable
change on a verified baseline** rather than a comparison between two different pipelines.

---

## Why the previous custom loop could not match nnU-Net

The old `train_kfold.py` / `src/training/trainer.py` path approximated nnU-Net with MONAI
transforms and a hand-written epoch. Each of these differences costs accuracy, and
together they explain why SGD at nnU-Net's own LR collapsed to all-background there:

| | old loop | nnU-Net (and the replica) |
|---|---|---|
| weight init | PyTorch default `kaiming_uniform(a=√5)` | `InitWeights_He(1e-2)` + residual blocks zero-initialised |
| patch sampling | `RandCropByPosNegLabeld(pos=1, neg=2)` | last 33 % of the batch forced onto a random foreground **class**; patches may hang off the volume and are padded |
| spatial aug | `RandAffined` on the final patch | rotate/scale inside a **rotation-inflated** [243, 270, 205] patch, then centre-crop |
| aug probabilities | noise 0.15, one gamma 0.15, blur per-sample | noise 0.1, gamma **twice** (inverted 0.1 + normal 0.3, `retain_stats`), blur/low-res roll again per channel at 0.5 |
| normalisation | MONAI `NormalizeIntensityd(nonzero=True)` on the full volume | crop to brain box first, z-score over `seg >= 0`, re-imposed after augmentation by `MaskTransform` |
| Dice smoothing | MONAI defaults | `smooth = 1e-5` in numerator and denominator, denominator clamped at 1e-8 |
| epoch | full pass over the dataset | exactly 250 optimiser steps, sampled with replacement |
| checkpoint rule | best full-image region Dice | best **EMA of the 50-patch pseudo-Dice** |
| schedule | poly **+ 50-epoch warmup**, or AdamW | poly `(1 - e/E)^0.9`, no warmup, SGD 1e-2 / 0.99 nesterov |
| inference | MONAI `sliding_window_inference` | nnU-Net step placement, Gaussian σ = patch/8, mirroring averaged over **logits** |

The weight initialisation and the missing forced-foreground *class* sampling are the two
most likely causes of the recorded SGD collapse; both are fixed here.

---

## Layout

```
src/nnunet_replica/
  plans.py          parse plans.json -> patch / batch / topology / batch_dice / DS scales
                    (reads both the >=2.4 `architecture` schema and the <=2.3 inline one)
  preprocessing.py  crop-to-nonzero -> masked z-score -> resample -> class locations
  dataloading.py    nnUNetDataLoader3D patch sampling, served as whole batches
  augmentation.py   nnU-Net's batchgenerators pipeline, transform for transform
  network.py        build from the plan + InitWeights_He(1e-2); SegResNetDS escape hatch
  loss.py           soft Dice (no bg) + CE, deep-supervision wrapper
  lr_scheduler.py   poly LR
  logger.py         per-epoch log, EMA pseudo-Dice, progress.png
  trainer.py        the loop: 250-iteration epochs, SGD, AMP, clip 12, EMA checkpointing
  inference.py      Gaussian sliding window + mirroring TTA
  splits.py         patient-level folds -> case-id lists

preprocess_replica.py   build the case cache
train_replica.py        train one fold
evaluate_replica.py     out-of-fold CV, or held-out test with folds ensembled
```

Only `batchgenerators` (augmentation) and `dynamic_network_architectures` (the U-Net
classes the plan names) are borrowed — both are libraries the native run also uses.
Nothing in the training path imports `nnunetv2`.

---

## Running it

```bash
# 1. Build the preprocessed case cache once (~10 min with 8 processes, ~30 GB at fp16).
#    Idempotent: re-running resumes. --fold N does only what that fold needs.
python preprocess_replica.py --config experiments/exp20_replica_resenc_m_11g_5fold.yaml \
    --num_processes 8

# 2. Train a fold.
python train_replica.py --config experiments/exp20_replica_resenc_m_11g_5fold.yaml --fold 0
python train_replica.py --config experiments/exp20_replica_resenc_m_11g_5fold.yaml --fold 0 --c   # resume

# 3. Score it.
python evaluate_replica.py --config experiments/exp20_replica_resenc_m_11g_5fold.yaml \
    --run_dirs runs/replica_*_fold0 --split val        # out-of-fold CV
python evaluate_replica.py --config experiments/exp20_replica_resenc_m_11g_5fold.yaml \
    --run_dirs runs/replica_*_fold[0-4] --split test   # held-out, 5 folds ensembled

# 4. Post-processing, the nnU-Net way: determine on OOF, apply to test.
python evaluate_replica.py --config <cfg> --run_dirs runs/replica_*_fold[0-4] \
    --split val --determine_postprocessing              # -> <out>/postprocessing.json
python evaluate_replica.py --config <cfg> --run_dirs runs/replica_*_fold[0-4] \
    --split test --postprocessing_json <out>/postprocessing.json
```

## Ensembling several models

`evaluate_replica.py` ensembles the 5 folds of one experiment. `evaluate_ensemble.py`
ensembles across experiments — the analogue of `nnUNetv2_ensemble` — by averaging the
members' softmax and argmaxing once at the end:

```bash
python evaluate_ensemble.py --split test \
    --member experiments/exp20_replica_resenc_m_11g_5fold.yaml runs/exp20_fold[0-4] \
    --member experiments/exp21_replica_5ch_subtraction_5fold.yaml runs/exp21_fold[0-4] \
    --member experiments/exp25_replica_segresnet_ds_5fold.yaml runs/exp25_fold[0-4]
```

Members may differ in architecture, plan, patch size and input channels — a 4-channel model
and a 5-channel subtraction model ensemble fine. `--weights` sets per-member weights (folds
inside a member are always averaged equally, so a member's influence does not depend on how
many folds it contributed). `--split val` gives the ensemble's out-of-fold CV number, with
each case predicted only by the fold that held it out; `--split test` uses every fold.
`--determine_postprocessing` and `--postprocessing_json` work exactly as they do for a single
model.

Two things it enforces rather than assumes, because both failures are silent:

* **Averaging happens in the original uncropped volume.** Members with different input
  channels get different nonzero-crop bboxes, so their cropped predictions are not
  voxel-aligned; each is un-cropped first, with the probability mass outside a member's crop
  put on background (what the crop asserted about that region anyway). A flat-zero fill there
  would let the loosest-cropping member decide the argmax on its own.
* **Every member must share the patient split.** Differing per-fold validation sets or
  held-out test ids are refused by name, since the ensemble would otherwise score a case with
  a network that trained on it. Differing *train* sets are fine and expected — exp27 adds
  synthetic cases to train only.

Weights live on the CPU between networks and move to the GPU per case, so a 3-member x 5-fold
line-up of 100M-parameter nets runs on the 8 GB card; the transfers are milliseconds against
seconds of 8-flip sliding-window inference.

Checks: `tools/replica_parity/ensemble_checks.py` (7 checks).

## Inference and post-processing

**Inference** is a port, not a delegation to MONAI, and is bit-exact against native 2.4.2:
Gaussian tile weighting (σ = patch/8, zeros lifted so the weight map can't divide by zero),
nnU-Net's evenly-spread step placement, and complete mirroring TTA — all
2³ = 8 flip combinations, with the **logits** averaged before a single softmax. Fold
ensembling averages softmax across folds, as `nnUNetv2_ensemble` does. The one nnU-Net step
not ported is resampling logits back to the original spacing before argmax; BraTS is already
at the plans' 1 mm isotropic target so it is a no-op, and `undo_cropping` raises if a plan
ever does resample rather than silently misaligning.

**Post-processing** — two different things, easy to conflate:

| | what it does | when |
|---|---|---|
| `--determine_postprocessing` / `--postprocessing_json` | nnU-Net's own: greedily accept "remove all but largest component" ops, first over all foreground merged (accepted only if the mean improves **and** no single label degrades), then per label (accepted only if **that label's** Dice improves). 26-connectivity; both-empty Dice is NaN and excluded. Frequently a no-op. | to report an nnU-Net-comparable number |
| `--postprocess` / `inference.postprocess` | BraTS-competition heuristic with hand-picked thresholds: small-component cleanup, ET < 250 vx folded into the core, hole filling. nnU-Net does **none** of this. | as a labelled comparison only |

Determination reads only out-of-fold validation predictions and writes the accepted ops to
`postprocessing.json`; that file is what gets applied to the held-out test set, so the test
split never influences its own post-processing. Predictions are cached to
`<out>/predictions_val/`, so determination costs no extra inference.

### Under Docker

```bash
docker-compose run --rm replica-preprocess                      # build the cache
REPLICA_FOLD=0 docker-compose run --rm replica-train            # 8 GB: recomputation on
REPLICA_FOLD=0 docker-compose run --rm replica-train-11g        # >=11 GB: full footprint
REPLICA_FOLD=0 docker-compose run --rm replica-train-11g-resume
docker-compose run --rm replica-fit-check-11g                   # does batch 2 fit? ~2 min
docker-compose run --rm replica-smoke                           # 2-epoch wiring check
RUN_DIRS="runs/replica_*_fold0" docker-compose run --rm replica-eval-val
```

Env vars: `REPLICA_CONFIG` (experiment basename), `REPLICA_FOLD` (0-4), `REPLICA_ARGS`
(extra `train_replica.py` flags), `RUN_DIRS` (for the eval services). Compose interpolates
these at parse time, so **export them or prefix the command** — `-e` will not work.

`replica-train-11g` differs from `replica-train` only by `--no_grad_checkpointing`: same
plan, same recipe, same resulting model, but activations are stored rather than recomputed
(~2x faster steps at the plan's full ~11 GB). It OOMs on the 8 GB laptop at the plan's
batch 2; `replica-fit-check-11g` tells you in ~2 minutes whether a given GPU can take it.

No image rebuild is needed for replica work — `src/`, `plans/`, and the three
`*_replica.py` scripts are bind-mounted, and the existing `brats-seg` image already
carries batchgenerators 0.25.3 / dynamic_network_architectures 0.4.4 / torch 2.4.1.

A run directory carries nnU-Net's own artefacts — `training_log_*.txt`, `progress.png`,
`checkpoint_best/latest/final.pth`, `validation_summary.json` — next to the repo's
`config.yaml` and TensorBoard logs.

---

## Measured on the 8 GB RTX 3070 Laptop (16 cores)

| | result |
|---|---|
| ResEnc-M, plan batch 2 @ [128, 192, 128], encoder grad-checkpointing | **7.07 GB, 1.09 s/step** |
| ResEnc-M, same, no checkpointing | OOM |
| ResEnc-M, batch 1, no checkpointing | 7.79 GB, 0.48 s/step |
| PlainConvUNet (exp24), batch 2, no checkpointing | OOM — needs `grad_checkpointing: true` |
| SegResNetDS (exp25), batch 2 | OOM — no encoder hook, needs `grad_accum_steps: 2` |
| augmentation, single worker | ~11.3 s/batch (SpatialTransform 6.9 s, SimulateLowResolution 3.2 s) |
| preprocessing | ~1.7 s/case single-threaded |

Two consequences worth planning around:

**Memory is fine; throughput is the constraint.** With `grad_checkpointing: true` the
*planned* batch 2 fits, so no gradient accumulation is needed and the run is a true
replica. But the augmentation is memory-bandwidth-bound on the inflated [243, 270, 205]
patch, and 12 workers do not scale linearly — measured epochs on this box run well above
the 273 s the GPU alone would need. Expect the data pipeline, not the GPU, to set the
epoch time here. On the 12 GB desktop that already runs the native training, set
`grad_checkpointing: false` and expect roughly nnU-Net's own epoch times.

**Gradient accumulation is exact when you need it.** `batch_dice` is `False` in this
plan, so Dice is a per-sample mean and CE a per-voxel mean; InstanceNorm never mixes
samples. N micro-steps of size B/N therefore produce *identical* gradients to one step of
size B — `grad_accum_steps: 2` is a substitute, not an approximation. The trainer warns
if you combine it with `batch_dice: true`, where the equivalence breaks.

---

## Parity with native nnU-Net

The claim this section supports is not "the replica trains to similar numbers" — it is
**the replica computes what nnU-Net computes**. Harness and reproduction steps:
[`tools/replica_parity/`](tools/replica_parity/README.md).

Setup: **Dataset500_ReplicaCmp**, a patient-grouped subset of exp20's fold 0 (161 train /
40 val), on the genuine 8 GB ResEnc-M plan — patch `[128, 160, 112]`, batch 2,
`batch_dice: False`, last stride `(2, 2, 1)`, 101,945,177 parameters. The native side is
nnU-Net **2.4.2** on `PYTHONPATH` rather than installed, so torch, batchgenerators and
dynamic_network_architectures are byte-identical on both sides — a library version
difference would invalidate the whole comparison.

**Component equivalence — 9/9, every difference exactly zero.**

| check | result |
|---|---|
| plan parsing | patch, batch, `batch_dice` identical |
| deep supervision scales + weights | max\|diff\| `0.000e+00`, weights `[0.5333, 0.2667, 0.1333, 0.0667, 0.0]` |
| network architecture + init | 101,945,177 params both; state_dict keys identical; init and forward max\|diff\| `0.000e+00` |
| Dice+CE loss (+ deep supervision) | `1.77445281` vs `1.77445281`, \|d\| `0.00e+00` |
| poly LR schedule | max\|diff\| `0.000e+00` |
| rotation config + inflated patch | `[224, 228, 189]` both; mirror axes `(0,1,2)` |
| augmentation pipeline | 14 transforms, list identical; **train data and target max\|diff\| `0.000e+00`** |
| patch sampler | 400 bbox draws, 0 mismatches; forced-fg pattern identical |
| preprocessing | 3 real cases: **data max\|diff\| `0.000e+00`**, seg identical, fg-location counts match |

**Step-level equivalence — identical weights, identical batches.** This is the decisive
test, because it removes the RNG and data-order confound that makes two independent runs
incomparable.

| check | result |
|---|---|
| `validation_step` | loss **exactly identical** (\|d\| `0.00e+00`); hard TP/FP/FN max\|diff\| `0/0/0` |
| `train_step` (AMP, grad-clip 12, SGD+Nesterov) | losses agree to ≤ `4.8e-7`; **weights after 3 steps differ by `0.000e+00` across 383,074,841 values** |
| inference (sliding window + Gaussian + mirroring TTA) | **99.9992–99.9994 % voxel agreement**; pred-vs-pred Dice ≥ 0.9991 |

**Speed — the loops are equally fast.** Timed on identical fixed batches, run sequentially:

| | s/step | 250-iter epoch |
|---|---|---|
| native, `torch.compile` ON (as shipped) | 0.621 | 155 s |
| native, `torch.compile` OFF | 0.711 | 178 s |
| replica (never compiles) | 0.699 | 175 s |

`replica ÷ native-without-compile = 0.98x` — the same loop, same speed. nnU-Net's edge is
`torch.compile` (1.15x), which the replica does not use. Note that the GPU alone needs only
~175 s per epoch while measured epochs run 341–528 s: **the data pipeline, not the loop,
sets epoch time on this box**, consistent with the augmentation cost measured above.

### Why the 2-epoch end-to-end numbers prove nothing on their own

Three replica runs — two of them at the *same seed* — against the native run, all on the
same 161/40 partition:

| run (epoch 1) | ET | mean fg | train loss | val loss |
|---|---|---|---|---|
| replica, seed 42 (run A) | 0.8194 | 0.5654 | −0.2305 | −0.3177 |
| replica, seed 42 (run B) | 0.6948 | 0.5419 | −0.2206 | −0.3157 |
| replica, seed 1234 | 0.7157 | 0.5173 | −0.1703 | −0.2628 |
| **native nnU-Net 2.4.2** | **0.7726** | **0.5126** | **−0.1834** | **−0.2762** |

**The native run lands inside the replica's own run-to-run range on ET, train loss and val
loss, and within 0.005 on mean fg pseudo-Dice.** The replica's spread against itself
(ET 0.125) is *larger* than its gap to native (ET 0.078). At epoch 0 the same-seed runs gave
ET pseudo-Dice of 0.741, 0.008 and 0.000 — the metric is essentially noise this early,
because a class the model has barely begun to predict swings between "found" and "missed".

Note these runs are nondeterministic *despite* the fixed seed: `cudnn.benchmark` algorithm
selection, AMP reduction order, and 12 asynchronous dataloader workers all vary run to run.

The practical rule: **a 2-epoch run cannot validate a training loop** — the noise floor is
wider than any effect worth detecting. Use it to confirm nothing is grossly broken, and rely
on the step-level checks above for the actual claim. Reproduce the spread with
`cmp_train_replica.py --seed N`.

---

## The experiment series (exp20+)

One-line index: [`experiments/README.md`](experiments/README.md).

Every experiment below runs the **same loop**; each changes exactly one thing against
exp20, so the series reads as a set of controlled ablations rather than a set of
differently-built pipelines.

| exp | change vs exp20 | question |
|---|---|---|
| **20** | — (the replica baseline) | does our loop reproduce the native nnU-Net numbers? |
| **21** | 5th input channel, T1Gd − T1 | is an explicit enhancement-subtraction map worth it? |
| **22** | Dice + **Focal** instead of Dice + CE | does focal recover the rare classes (NETC, ET)? |
| **23** | 5 channels **and** focal | are 21 and 22 additive or redundant? (completes the 2×2) |
| **24** | **PlainConvUNet** plan (`nnUNetPlainUNet_11GBPlans.json`) | is the residual encoder worth 3.3× the parameters? |
| **25** | **SegResNetDS** network | architecture comparison under an otherwise identical recipe |
| **26** | 250 epochs instead of 1000 | what does a 4× shorter schedule actually cost? |

Notes that matter when reporting:

* exp24 is an unusually clean architecture test — the default planner independently chose
  the *same* patch [128, 192, 128] and batch 2, so ResEnc (102.4 M) vs PlainConv (31.2 M)
  is the only difference. This is the comparison exp18-vs-exp20 previously tried to make
  and could not, because those two also differed in engine and recipe.
* exp25 runs the **identical loop** — same sampler, augmentation, optimiser, schedule,
  epoch definition, checkpoint rule and inference. Deep supervision also lines up exactly:
  SegResNetDS emits 4 heads (1, 1/2, 1/4, 1/8) where the plan has 5, but the missing one is
  precisely the head nnU-Net zeroes, so the trainer drops the zeroing and both end up
  training the same four resolutions with weights `[0.5333, 0.2667, 0.1333, 0.0667]`
  (logged at startup in every run). Two things genuinely differ: the network, and its
  initialisation — SegResNetDS keeps MONAI's default rather than `InitWeights_He(1e-2)`,
  since He-initialising a different residual design is a change in its own right.
* exp26 exists so the rest of the series is affordable. Compare exp26 only against other
  250-epoch runs — pitting a 250-epoch ablation against the 1000-epoch exp20 baseline
  would confound the budget with the change under test.
* exp21/exp23 need `*-sub.nii.gz` precomputed (`precompute_subtraction.py`) and use a
  **separate** 5-channel cache. exp20/22/24/25/26 all share the 4-channel one.

Fewer parameters does not mean less memory: activations dominate on an 8 GB card, so
exp24's 31 M-parameter PlainConvUNet still needs recomputation at batch 2 (measured, not
assumed). SegResNetDS has no encoder-stage hook for the checkpointing helper, so exp25
uses exact gradient accumulation instead.

The pre-replica exp20–exp25 configs moved to `experiments/superseded/` so the numbering
is unambiguous; they still describe the runs already in `runs/`, and the `exp20*`
docker-compose services still point at them. New work should use the `*_replica_*`
configs.

---

## Data hygiene (unchanged from the rest of the repo)

Patient-level splitting with `split_seed: 42`; no BraTS-GLI patient appears in two splits.
With `data.kfold_holdout_test: true` (the default for the replica configs) the seed-42
10 % test patients — 74 patients / 153 cases — are held out of **every** fold, exactly as
the native exp19 run did, so `--split test` is directly comparable to the native
ensemble. `--split val` is the out-of-fold CV number, which is what the native
`nnUNetv2_train` validation reports.
