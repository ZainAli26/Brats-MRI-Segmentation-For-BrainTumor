# LOCAL_RUN_STATE.md — exp20 vs exp22 loss comparison (local 3070 box)

Resume file for the **local Linux box** (RTX 3070 Laptop 8 GB), analogous to
`RUNPOD_STATE.md` for rented boxes. If the power cuts, a session dies, or you come back
in a week: **read this file first, then run the resume command in §5.**

Last updated: 2026-08-14, during exp20 fold 0 training.

---

## 1. The goal

Compare **loss functions**: Dice-CE (exp20) vs Dice-Focal (exp22), **both at 250 epochs**,
fold 0, 4-channel. Then produce a full analysis, not just a winner.

The two configs differ in **exactly two lines** from their 1000-epoch parents
(`num_epochs`, `model.name`), and from *each other* only in:

```
loss: "dice_ce"   ->  loss: "dice_focal"  +  focal_gamma: 2.0
model.name        ->  ..._dicefocal_250ep
```

Everything else — plan, patch [128,192,128], batch 2, SGD lr 0.01 / momentum 0.99 /
poly^0.9, seed 42, fold 0, `save_every: 50`, deep supervision, augmentation, the cache —
is identical. Loss is the single free variable.

**exp23 (5-channel) is parked**, not cancelled. It answers a different question (the
subtraction channel), and its config `exp23_replica_5ch_dicefocal_250ep_5fold.yaml` is
already written and its cache already built if you want it later.

---

## 2. Why 250 and not exp20's native 1000

exp20 at 1000 epochs is ~15 days per fold on this GPU. More importantly, scoring a
250-epoch exp22 against a 1000-epoch exp20 is invalid — poly LR annealing, not the loss,
would dominate the gap (see CLAUDE.md "Comparison rules"). Both arms were cut to 250 with
the identical transform, so the comparison stays honest and costs ~7.4 days instead of ~30.

Deliberately **not** exp26, even though exp26 is also exp20@250: exp26 sets
`save_every: 25` vs exp22's 50 and is its own experiment about budget. Deriving both arms
here keeps every knob but the loss identical.

---

## 3. Environment facts that will bite you

| fact | consequence |
|---|---|
| `runs/` and `.cache/` are **root-owned** (created by earlier Docker runs) | a **native** `python3 train_replica.py` dies with `PermissionError` creating its run dir. There is **no passwordless sudo** on this box. |
| The replica caches are **not** in `./.cache/replica` — they are at **`/home/zain/brats_replica_cache/`** | compose bind-mounts them via `REPLICA_CACHE`; configs keep the parent's `/app/.cache/replica/...` path |
| Docker is the sane path | container runs as root, so it writes into `runs/` normally. `docker-compose` is **v5.1.1** (the `docker-compose` command, *not* `docker compose`). |
| Image `brats-seg`: torch 2.4.1+cu124, monai 1.5.2, CUDA True | host native env is torch 2.10.0+cu128 — **different**. Both arms must run in the SAME env, i.e. both in Docker. |
| compose live-mounts `./src`, `./experiments`, `./plans`, `./train_replica.py` etc. | code edits need no image rebuild |
| host `.venv` is stale (no torch) | never use it |

### Caches (both built and verified)

| cache | path | contents | size |
|---|---|---|---|
| `Dataset102_4ch` | `/home/zain/brats_replica_cache/Dataset102_4ch` | 4863 files = **1621 × 3**, verified | 46 GB |
| `Dataset102_5ch` | `/home/zain/brats_replica_cache/Dataset102_5ch` | 9267 files = 3089 × 3 (1621 real + 1468 synthetic regaug) | 110 GB |

The 5ch cache holds exp27's synthetic cases. That is **safe** for exp23: `trainer.py:181`
builds `PreprocessedDataset(folder, self.train_case_ids)` — cases come from the split id
list, never from globbing the folder — and exp23 declares no `synthetic_train_dirs`.

Disk after both caches: 144 GB free of 929 GB.

---

## 4. Splits (identical for both arms, seed 42)

```
Fold 0: train 526 patients / 1163 cases,  val 131 patients / 305 cases
Test (held out): 74 patients / 153 cases  — reserved out of every fold
Total 731 patients / 1621 cases
```

---

## 5. THE RESUME COMMAND

Always `cd /home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/brats-segmentation` first.

```bash
export REPLICA_CACHE=/home/zain/brats_replica_cache

# --- resume exp20 (Dice-CE arm) ---
REPLICA_CACHE=/home/zain/brats_replica_cache \
REPLICA_CONFIG=exp20_replica_resenc_m_11g_250ep_5fold.yaml \
REPLICA_FOLD=0 REPLICA_ARGS="--num_workers 14" \
  docker-compose run --rm -T replica-train-resume

# --- resume exp22 (Dice-Focal arm) ---
REPLICA_CACHE=/home/zain/brats_replica_cache \
REPLICA_CONFIG=exp22_replica_dicefocal_250ep_5fold.yaml \
REPLICA_FOLD=0 REPLICA_ARGS="--num_workers 14" \
  docker-compose run --rm -T replica-train-resume
```

`replica-train-resume` is `replica-train` + `--c`. It restores optimizer, GradScaler, EMA
and history, and continues **in place** in the existing run dir. To start fresh instead,
use service `replica-train` (no `--c`).

Run detached with a host-side log:

```bash
nohup env REPLICA_CACHE=... REPLICA_CONFIG=... REPLICA_FOLD=0 REPLICA_ARGS="--num_workers 14" \
  docker-compose run --rm -T replica-train-resume > logs/<name>.log 2>&1 &
```

**Checkpoint caveat (CLAUDE.md):** `trainer.py:422` is a bare `torch.save`, so checkpoints
are **not atomic**. A power cut mid-save can leave a truncated `checkpoint_latest.pth`.
`save_every: 50` means you lose at most 50 epochs (~18 h) — if `checkpoint_latest.pth` is
corrupt, fall back to `checkpoint_best.pth` via `--resume_dir`.

---

## 6. Status board — UPDATE THIS AS RUNS FINISH

| # | run | config | status | run dir |
|---|---|---|---|---|
| 1 | 4ch cache | — | **DONE** (4863 files, 46 GB, 11m33s) | `/home/zain/brats_replica_cache/Dataset102_4ch` |
| 2 | exp20 fold 0 @250 (Dice-CE) | `exp20_replica_resenc_m_11g_250ep_5fold.yaml` | **RUNNING** (started 2026-08-14 ~05:13 container time) | `runs/replica_nnunet_replica_resenc_m_11g_250ep_*_fold0` |
| 3 | exp22 fold 0 @250 (Dice-Focal) | `exp22_replica_dicefocal_250ep_5fold.yaml` | QUEUED — start when #2 finishes | — |
| 4 | evaluation + analysis | see §7 | QUEUED | — |

Runs are **sequential**: 8 GB fits one ResEnc-M at a time. Do not start #3 while #2 runs.

**Cost — MEASURED on this exact run (2026-08-14), not estimated:**

| epoch | duration | implies 250 epochs |
|---|---|---|
| 0 | 942.0 s | 65.4 h = 2.73 days |
| 1 | 1010.9 s | 70.2 h = 2.93 days |

So **~2.9 days per run, ~5.8 days for both** at `--num_workers 14` in Docker. This is
faster than the ~1,280 s/epoch of the 2026-08-13 regaug run, which was 5-channel and
shared the box with other work. Sanity: `mean_fg_dice` 0.303 (epoch 0) → 0.510 (epoch 1),
so it is learning normally.

Re-measure any time from the `epoch_start_timestamps` / `epoch_end_timestamps` pairs in
the run's `training.log`.

**Dead artifact:** `runs_local/replica_nnunet_replica_resenc_m_11g_250ep_20260814_100032_fold0`
is an abandoned ~15-minute native-env attempt from before the Docker switch. Ignore it; it
is not part of the comparison. (Left in place rather than deleted.)

---

## 7. BraTS evaluation plan (both arms, identical treatment)

**One command, after the GPU is free:**

```bash
nohup ./run_brats_eval.sh > logs/brats_eval.log 2>&1 &
```

It refuses to start if the GPU has >1000 MiB in use, so it cannot collide with training.
It runs **four passes per arm** — val raw, val + BraTS heuristic, test raw, test + BraTS
heuristic. Predictions cache to `<run_dir>/predictions_<split>/`, so the post-processed
passes reuse the raw pass's inference and cost only metrics.

Outputs land in each run dir as `replica_metrics_*.csv` and `replica_summary_*.json`.

**Two post-processing families — never conflate them (CLAUDE.md, [[postprocessing-two-families]]):**

| flag | family |
|---|---|
| `--postprocess` | **BraTS-competition heuristic**, hand-picked thresholds: small-component cleanup, ET<250vx→core, hole filling. nnU-Net does none of this. This is "the BraTS eval". |
| `--determine_postprocessing` / `--postprocessing_json` | nnU-Net's own data-driven keep-largest-component ops, determined on OOF val and applied to test. A separate question; **never determine on test**. |

`run_brats_eval.sh` runs the heuristic family only.

With only fold 0 trained, `--split test` is a **single model**, not a 5-fold ensemble —
fine, because both arms are treated identically, but it is not comparable to native 5-fold
ensemble numbers. Say so when reporting.

**Already known without any extra inference** — the trainer's own end-of-run full-image
validation (sliding window + mirroring TTA, 305 val cases) is in each run's `summary.json`:

| | exp20 (Dice-CE) | exp22 (Dice-Focal) |
|---|---|---|
| best EMA pseudo-Dice | 0.8726 | pending |
| ET | 0.6755 | pending |
| TC | 0.6658 | pending |
| WT | 0.9115 | pending |
| RC | 0.6875 | pending |
| mean region Dice | 0.7351 | pending |

Report ET / TC / WT / RC Dice + HD95. **Watch NETC and ET specifically** — focal exists to
help the rare classes (NETC is ~0.004 % of voxels), so a fair verdict looks at whether it
moved the rare classes even if the mean is flat.

Do **not** turn on `--postprocess` or `--determine_postprocessing` for the headline
comparison, and never determine post-processing on the test split (CLAUDE.md).

---

## 8. If you must start over from nothing

1. Caches already exist (§3) — do not rebuild. If one is missing:
   `REPLICA_CACHE=... REPLICA_CONFIG=<cfg> docker-compose run --rm -T replica-preprocess`
   (~12 min for 4ch, CPU-only, idempotent).
2. Launch §5 with service `replica-train` (no `--c`).
3. Never edit the parent `exp20/exp22/exp23_*_5fold.yaml` configs — the 250-epoch
   derivatives are separate files by design.
