# RunPod RTX 4090 — end-to-end setup and run guide

Everything needed to take a fresh RunPod 4090 from empty to a finished exp20–27 series
plus one native nnU-Net run. Written 2026-08-13.

Companion docs: `gcs_download_commands.txt` (data pull), `NNUNET_REPLICA.md` (what the
replica loop is), `experiments/EXPERIMENT_PLAN.md` (what each experiment isolates),
`exp27_commands.txt` (registration augmentation).

**Default for every replica run in this guide: the 11 GB ResEnc-M plan
(`plans/nnUNetResEncM_11GBPlans.json`) with `--no_grad_checkpointing`** — patch
[128,192,128], batch 2, activations stored not recomputed. That is what the `-11g`
compose services do. It is ~2× faster per step than the checkpointed 8 GB-card path and
is the footprint the plan was actually budgeted for.

---

## 0. Read this first — three things that will bite you

**1. On RunPod you do not run Docker; the pod *is* the container.** A standard pod is
itself a container on a shared host, so `docker build` / `docker-compose` inside it
generally will not work (no daemon, not privileged). Verify on your pod with
`docker info` — if it errors, use Path A or B in section 2 rather than fighting it.
The consequence: **`docker-compose.yml` is a reference, not a runner.** Sections 6–9 give
the plain `python` equivalent of every compose service you need.

**2. `/dev/shm` must be large or the dataloader dies.** `docker-compose.yml:52` sets
`shm_size: "8g"` for a reason — 12 persistent workers share tensors through it. RunPod
pods often ship a 64 MB `/dev/shm`, which surfaces as `DataLoader worker (pid N) is
killed by signal: Bus error` about a minute into training. Check and fix in section 3.

**3. exp27's synthetic path is not overridable from the CLI.** `--data_dir` and
`--extra_data_dir` override the real dirs, but `data.synthetic_train_dirs` is read from
the YAML only (`src/nnunet_replica/splits.py:70`) and is the literal string
`../Brats2024/synthetic_regaug`. Your directory layout has to make that resolve — see
section 4. Everything else is CLI-overridable.

---

## 1. Pod specification

| setting | value | why |
|---|---|---|
| GPU | 1 × RTX 4090 24 GB | the plan fixes batch 2 / patch [128,192,128]; VRAM above 24 GB is unspendable here |
| vCPU | **24–32; 16 is the floor** | derived below — this is the binding constraint, not the GPU |
| RAM | **1.5 GB × vCPU + 16 GB** → 64 GB at 16 vCPU, 96–128 GB at 32 | measured ~1 GB working set per worker |
| Container disk | 30 GB | image + OS only |
| **Network volume** | see the disk table below | persists across pods; region-locked |
| Region | pick one and stay in it | a new region means re-downloading 88 GB |

**Avoid L4 and A10G** even if cheaper — L4's 300 GB/s is *below* the laptop 3070's
448 GB/s, and 3D convs here are bandwidth-bound. You would pay cloud rates for laptop
throughput.

### Why vCPU is the constraint — and why a faster GPU makes it worse

From `exp20_replica_resenc_m_11g_5fold.yaml:109`: augmentation costs **~11 s per batch,
single-threaded** (the `SpatialTransform` on the rotation-inflated [243,270,205] patch
dominates). So the workers needed to keep the GPU fed is just `11 s ÷ step_time`:

| GPU / mode | step time | workers to keep it fed |
|---|---|---|
| 3070 8 GB, checkpointed | 1.09 s (measured) | ~11 → hence the config's `num_workers: 12` |
| 3070, no checkpointing | ~0.55 s | ~20 |
| **4090, no checkpointing** | **~0.2–0.4 s (est.)** | **~28–55** |

**The config's 12 workers was tuned for a slow, checkpointed 3070 step.** Dropping
gradient checkpointing on a 4090 makes the GPU step ~5× shorter, so the CPU requirement
goes *up* ~5×. No typical RunPod 4090 offer provides 30+ vCPU, which means **you should
expect to be loader-bound no matter what you rent** — the question is only by how much.

Practical consequence: **shop for vCPU-per-GPU, not for the GPU.** Between two 4090 offers
at the same price, the one with more vCPU will finish sooner. Then set
`--num_workers $(nproc) - 2`.

| vCPU | `--num_workers` | roughly |
|---|---|---|
| 8 | 6 | badly starved; ~half the 3070's *checkpointed* throughput advantage wasted |
| 16 | 14 | usable floor — call it ~1.5–2× the 3070 in practice |
| 24–32 | 22–30 | the sweet spot; close to feeding the card |
| 48+ | 30 | wasted — you become GPU-bound, which is the goal |

The ~11 s figure was measured on this box's cores, and cloud per-core speed differs, so
treat the table as a sizing guide and let section 5's ten-epoch measurement give you the
real number.

**RAM follows from the worker count**, not from the dataset: `dataloading.py:197` records a
**~1 GB working set per worker** (which is why `prefetch_factor` is pinned at 1). Each run
also spawns `num_workers//2` validation workers that stay resident, so budget
`1.5 GB × num_workers`, plus ~16 GB for the main process and page cache.

### Disk budget

| item | size | needed for |
|---|---|---|
| Docker image | ~15–20 GB | always |
| Largest tarball, transient during pull | +43 GB peak | always |
| `training_data1_v2` + `_additional` | ~50 GB | always |
| `.cache/replica/Dataset102_4ch` | ~46 GB | exp20/22/24/25/26 |
| `.cache/replica/Dataset102_5ch` | ~57 GB | exp21/23/27 |
| `Brats2024/synthetic_regaug` | ~40 GB | exp27 only |
| ...plus synthetic cases in the 5ch cache | ~34 GB | exp27 only |
| `nnunet_data/` (raw + preprocessed + results) | ~120 GB | native run only |
| `runs/` checkpoints, whole series | ~60 GB | always (1.2 GB × 3 per fold) |

Which gives three sensible volume sizes:

| scope | sum | provision |
|---|---|---|
| Tiers 1–3, no exp27, no native | ~235 GB | **300 GB** |
| + exp27 | ~310 GB | **400 GB** |
| + native nnU-Net | ~430 GB | **500 GB** |

Only the 4ch cache figure is measured (29 MB/case × 1621); the 5ch numbers scale it by 5/4
and the nnU-Net native figure is an estimate — hence the headroom in each tier. Note the
transient tarball line assumes the section-4 pull loop that deletes each tar right after
its own extract; downloading all three first needs 88 GB instead of 43 GB.

Network volume storage bills continuously whether or not a pod is attached — that is the
point (it saves the ~$10.50 GCS egress on every re-rent) but budget for it and delete the
volume when the series is done.

---

## 2. Getting the container onto the pod

### Path A — build locally, push to a registry (recommended)

Reproducible, and the pod boots ready to train. Do this on **your local box**:

```bash
cd <repo>/brats-segmentation

# .dockerignore does not exclude .cache/ — make sure it is not in the build context
echo '.cache/' >> .dockerignore

docker build -t <dockerhub-user>/brats-seg:1.0 .
docker login
docker push <dockerhub-user>/brats-seg:1.0
```

Then in the RunPod UI: **Templates → New Template**, container image
`<dockerhub-user>/brats-seg:1.0`, volume mount path `/workspace`, expose port 22 (SSH)
and 6006 (TensorBoard). Deploy a 4090 pod from that template.

The image is ~15–20 GB, so the first pod start pulls for several minutes. Subsequent pods
in the same region are usually cached.

### Path B — official PyTorch template + pip (faster to start, less reproducible)

**RunPod's PyTorch 2.8.0 template works.** Any 2.4–2.12 template does. The local box runs
torch 2.10.0+cu128 and the Windows box runs 2.12+cu126, so 2.8 is comfortably inside the
validated range, and the replica loop already uses the APIs that shift between versions:
`torch.amp.GradScaler("cuda")` (`trainer.py:173`), `checkpoint(..., use_reentrant=False)`
(`network.py:70`), and — the one that matters — an explicit
`torch.load(..., weights_only=False)` (`trainer.py:440`). Without that last one, torch
≥ 2.6's flipped default would break every `--c` resume.

**The risk is pip clobbering the template's torch, not torch itself.** `requirements.txt`
lists `torch>=2.1.0`; pip will normally leave a satisfying 2.8.0 alone, but a resolver
detour through `monai[all]` can replace the CUDA build with a generic or CPU-only wheel.
Install without the torch line and verify:

```bash
cd /workspace
git clone https://github.com/ZainAli26/Brats-MRI-Segmentation-For-BrainTumor.git
cd Brats-MRI-Segmentation-For-BrainTumor/brats-segmentation

# keep the template's CUDA torch; skip antspyx unless generating synthetic data on the box
grep -vE '^(torch|antspyx)\b' requirements.txt > /tmp/req.txt
pip install -r /tmp/req.txt

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

That must still print a `+cu12x` build with `True`. If torch got downgraded or replaced
with a CPU wheel, reinstall it from the CUDA index before doing anything else.

`antspyx` is only needed to *generate* synthetic data. The `synthetic_regaug` tarball is
pre-made, so skip it — it is a heavy wheel and its `import ants` is fragile.

**Path B changes the paths.** The configs hardcode `replica.preprocessed_dir:
/app/.cache/replica/...`. Outside the container there is no `/app`, so either
`ln -s /workspace/.../brats-segmentation /app`, or pass
`--output_dir` / `--preprocessed_dir` on every command. The symlink is one line and keeps
every command in this guide correct — do that.

---

## 3. First-boot checks on the pod

```bash
nvidia-smi                       # 4090, 24564 MiB, driver 550+
nproc                            # vCPU count — you need this for --num_workers
free -g                          # RAM
df -h /workspace                 # volume mounted and sized
df -h /dev/shm                   # MUST be >= 8G
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `/dev/shm` is 64 MB:

```bash
mount -o remount,size=16G /dev/shm      # works if the pod has CAP_SYS_ADMIN
df -h /dev/shm                          # confirm
```

If the remount is refused, you cannot use 12 workers. Fall back to `--num_workers 4` and
accept a slower, loader-starved run — and note it in the run log, because it changes epoch
times but not results.

---

## 4. Data layout — get this right before downloading

Two rules drive the whole layout:

- `data.synthetic_train_dirs` is the literal string `../Brats2024/synthetic_regaug`,
  resolved from the working directory. Not overridable (section 0, note 3).
- `replica.preprocessed_dir` is the literal `/app/.cache/replica/...` in every config.

Both are satisfied by making `/app` the working directory and `/Brats2024` the data root.

**Path A (image, `WORKDIR=/app`).** The repo is baked into the image; only data and
outputs live on the volume. Extract the tarballs to `/workspace/Brats2024`, then link:

```bash
ln -sfn /workspace/Brats2024 /Brats2024          # makes ../Brats2024 resolve from /app
mkdir -p /workspace/{runs,nnunet_data} /app/.cache
ln -sfn /workspace/runs        /app/runs
ln -sfn /workspace/nnunet_data /app/nnunet_data
```

Without the `runs` and `nnunet_data` links your checkpoints vanish when the pod is
terminated. Note `.cache` is deliberately a **real directory on container disk**, not a link
to the volume — see the next subsection for why.

### Network volume vs container disk — where the cache belongs

**The network volume is the right home for the raw data, tarballs, checkpoints and logs.**
Those are written once and read sequentially, and they must survive pod termination. Use it
for all of that.

**The preprocessed cache is different, because it is the hot path.** `dataloading.py:59`
memory-maps each case with `mmap_mode="r"` so that "only the sampled patch is faulted in
from disk" — and the sampled region is the inflated [243,270,205] bbox, which is *not*
contiguous inside the array. Every iteration is therefore hundreds of small strided
page-faults, the access pattern network-attached storage is worst at (latency per op, not
just bandwidth).

Per training iteration, at fp16:

| | per case | batch of 2 | at 0.3 s/step |
|---|---|---|---|
| 4-channel | ~121 MB | ~242 MB | **~800 MB/s** |
| 5-channel | ~148 MB | ~296 MB | **~1.0 GB/s** |

On the 3070 at 1.09 s/step this was only ~220 MB/s and local SSD absorbed it. Removing
gradient checkpointing on a 4090 shortens the step ~5×, so the storage demand rises the
same way the CPU demand does.

**The page cache is what saves you — if RAM is big enough.** After the first epoch the
whole cache can live in RAM and the volume stops mattering:

| experiments | cache size | verdict |
|---|---|---|
| exp20/22/24/25/26 (4ch) | ~46 GB | fits in 64 GB RAM → **network volume is fine** after warmup |
| exp21/23 (5ch) | ~57 GB | tight at 64 GB; comfortable at 96 GB+ |
| exp27 (5ch + synthetic) | ~91 GB | **does not fit 64 GB** → put the cache on local disk, or take 128 GB RAM |

### The recommended split: regenerable on container disk, irreplaceable on the volume

Do not think of it as "train locally, back up to the volume later". Split by **whether the
file can be regenerated**, and write each one to its final home from the start:

| path | disk | why |
|---|---|---|
| `/app/.cache/replica/*` | **container** | hot random-access path; regenerable in 20–30 min |
| `nnunet_preprocessed/` | **container** | same — nnU-Net rebuilds it from raw |
| `/app/runs/` (checkpoints, logs) | **volume** | irreplaceable; 1.2 GB sequential write every `save_every` epochs — no perf cost |
| `Brats2024/` raw NIfTI | **volume** | read once during preprocessing, sequentially; no need to duplicate 90 GB locally |
| `nnunet_results/` | **volume** | irreplaceable |
| tarballs | **volume** | keeping them saves ~$10.50 of GCS egress on the next pod |

```bash
mkdir -p /app/.cache                             # container disk — NOT the volume
ln -sfn /workspace/runs        /app/runs         # volume
ln -sfn /workspace/nnunet_data /app/nnunet_data  # volume (results); see note on preprocessed
ln -sfn /workspace/Brats2024   /Brats2024        # volume
```

**Why checkpoints must go straight to the volume rather than be rsynced there:**
`trainer.py:422` calls `torch.save` directly onto the final path — there is **no
write-to-temp-then-rename**. Two consequences:

- An rsync that fires while a 1.2 GB checkpoint is being written copies a **torn file**, and
  you would not find out until a resume failed. If you do rsync anything, exclude
  recently-modified files: `find … -mmin +5`.
- If the pod dies *mid-write*, `checkpoint_latest.pth` itself is left truncated. Your
  fallback is `checkpoint_best.pth`, which is a separate file — but potentially many epochs
  behind. On spot, set `replica.save_every: 10` and consider keeping a rotating copy.

Writing directly to the volume removes both problems and costs nothing measurable: it is one
sequential 1.2 GB write every 10–50 epochs, not a per-iteration path.

The only cost of keeping the cache on container disk is rebuilding it on each new pod —
measured at 0.67 s/case, so ~20–30 min for 1621 cases at 8 processes, while the GPU is idle
anyway. Cheap insurance against starving the card for days.

**Sizing that split:** container disk needs ~260 GB (image 20 + replica caches 137 +
`nnunet_preprocessed` ~100); the volume needs ~250 GB (raw data 90 + tarballs 88 + runs 60
+ nnunet_results 10). A 500 GB volume is over-provisioned unless you also intend to back up
the regenerable caches — which is paying continuous storage rent to avoid a 25-minute
rebuild.

**Settle it by measuring, not guessing** — one minute, before you build anything:

```bash
apt-get install -y fio
fio --name=r --rw=randread --bs=4k --size=2G --numjobs=4 --iodepth=16 \
    --direct=1 --runtime=30 --time_based --group_reporting --directory=/workspace
# repeat with --directory=/  (container disk) and compare IOPS and mean latency
```

If the volume's random-read numbers are within ~2× of container disk, keep everything on
the volume and move on.

**Path B (pip, no image).** Clone into the volume and mirror the local layout, which
already satisfies both rules:

```
/workspace/Brats-MRI-Segmentation-For-BrainTumor/
├── Brats2024/                      <- extract the tarballs HERE
│   ├── training_data1_v2/
│   ├── training_data_additional/
│   └── synthetic_regaug/           <- exp27 only
└── brats-segmentation/             <- cwd for every command
    ├── .cache/replica/             <- preprocessed caches land here
    ├── runs/                       <- checkpoints and logs
    └── nnunet_data/                <- native nnU-Net only
```

then `ln -s /workspace/Brats-MRI-Segmentation-For-BrainTumor/brats-segmentation /app` so
the hardcoded `/app/.cache/replica` paths resolve. `Brats2024/` is gitignored, so it sits
inside the clone safely.

**Sanity-check the links before preprocessing** — a wrong one costs you a 30-minute
cache build in the wrong place:

```bash
cd /app && ls ../Brats2024/ && readlink -f .cache runs
```

### Download the data

Follow `gcs_download_commands.txt` — auth with a read-only service-account key, then:

```bash
export BUCKET=gs://<your-bucket>
export DEST=/workspace                                        # Path A
# export DEST=/workspace/Brats-MRI-Segmentation-For-BrainTumor  # Path B
mkdir -p "$DEST/tar"

# detached, deletes each tar right after its own extract to cap peak disk
nohup bash -c '
  for f in training_data1_v2 training_data_additional synthetic_regaug; do
    gcloud storage cp "$BUCKET/tar/$f.tar" "$DEST/tar/" &&
    tar -xf "$DEST/tar/$f.tar" -C "$DEST" &&
    rm -f "$DEST/tar/$f.tar"
  done' > /workspace/pull.log 2>&1 &
```

Extract into the **parent** (`$DEST`), not `$DEST/Brats2024` — tar members already carry
the `Brats2024/` prefix. Drop `synthetic_regaug` from the loop if you are not running
exp27; it is 40 of the 88 GB.

Verify: 8100 / 1626 / 8808 `.nii.gz` files respectively (1350 / 271 / 1468 cases x 6 files).

---

## 5. Benchmark before committing to the plan — a ~$1 check

The entire cost model rests on an **estimated** 3× speedup over the 3070 that has never
been measured on cloud hardware. Ten minutes here de-risks the whole budget.

```bash
cd /app          # both paths, thanks to the symlinks in section 4

# 1. Does batch 2 fit without checkpointing? ~2 minutes, answers before a 40-min surprise.
python train_replica.py \
  --config experiments/exp20_replica_resenc_m_11g_5fold.yaml \
  --data_dir ../Brats2024/training_data1_v2 \
  --extra_data_dir ../Brats2024/training_data_additional \
  --fold 0 --no_grad_checkpointing --smoke_test --max_cases 4 \
  --num_workers 4 --no_final_validation
```

Expect ~11 GB peak. **Watch it with `nvidia-smi`, not
`torch.cuda.max_memory_allocated()`** — the latter under-reported by 1.8 GB on the same
config locally because it excludes the CUDA context, cuDNN workspaces and reserved
allocator blocks.

Then, once the cache exists (section 6), run ten real epochs and read two numbers:

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5
```

- **s/epoch** — the plan assumes ~48 s. Multiply out before booking days of compute.
- **GPU utilization** — locally the loop spent ~85% of wall time waiting on CPU-side
  augmentation (6.8 s/it actual vs a 1.10 s/it GPU floor). If utilization sits below ~60%
  on the 4090 you are loader-starved: raise `--num_workers` toward `nproc - 2` first, and
  only then consider a second concurrent run.

Set `--num_workers` from the pod's actual `nproc`. The configs ask for 12 (train) and
spawn `num_workers//2` more for validation — **18 persistent processes per run**.

---

## 6. Build the preprocessed caches

Two caches, shared across experiments. Build only what the tier you are running needs.

```bash
# 4-channel — exp20, exp22, exp24, exp25, exp26   (~46 GB, ~20-30 min at 8 processes)
python preprocess_replica.py \
  --config experiments/exp20_replica_resenc_m_11g_5fold.yaml \
  --data_dir ../Brats2024/training_data1_v2 \
  --extra_data_dir ../Brats2024/training_data_additional \
  --num_processes 8
```

For the 5-channel experiments (exp21, exp23, exp27) the subtraction channel must exist on
disk as `<case>-sub.nii.gz` **before** preprocessing. **It already does** — all 1621 real
cases and all 1468 synthetic cases carry it, and those files are inside the GCS tarballs
(verified 2026-08-13: the `-sub` files are 7.8 GB of `training_data1_v2`'s 42.6 GiB tar).
So there is normally nothing to generate. Just confirm:

```bash
find ../Brats2024/training_data1_v2 -name '*-sub.nii.gz' | wc -l        # expect 1350
find ../Brats2024/training_data_additional -name '*-sub.nii.gz' | wc -l # expect 271
find ../Brats2024/synthetic_regaug -name '*-sub.nii.gz' | wc -l         # expect 1468
```

Only if a count comes back short — a partial extract, or data pulled from somewhere other
than these tarballs — regenerate. It is cheap, idempotent, and works on **all three** dirs
including the synthetic one:

```bash
python precompute_subtraction.py --data_dir ../Brats2024/training_data1_v2
python precompute_subtraction.py --data_dir ../Brats2024/training_data_additional
python precompute_subtraction.py --data_dir ../Brats2024/synthetic_regaug
```

That last one is safe because `augment_registration.py:488` calls the very same
`compute_subtraction(warped_t1c, warped_t1n, raw=False)` during generation — running the
standalone script over already-warped synthetic cases reproduces it, so there is no need
to re-run the ~5.5 h registration pass just to recover a missing channel.

Do **not** pass `--raw`. The default z-scores each volume inside the brain mask before
subtracting; a raw `t1c - t1n` reflects arbitrary scanner scale mismatch rather than
contrast enhancement, and would differ from what every existing case was built with.

```bash
# 5-channel cache — exp21, exp23   (~57 GB)
python preprocess_replica.py \
  --config experiments/exp21_replica_5ch_subtraction_5fold.yaml \
  --data_dir ../Brats2024/training_data1_v2 \
  --extra_data_dir ../Brats2024/training_data_additional \
  --num_processes 8

# exp27 adds the synthetic cases to that same 5ch cache (+~34 GB).
# NOTE: synthetic_regaug is deliberately NOT on the command line — there is no
# --synthetic_data_dir flag. It comes from data.synthetic_train_dirs in the YAML, and
# build_replica_splits attaches those cases to every fold's train list (splits.py:183)
# BEFORE the preprocessor's case list is built (splits.py:187), so they do get cached.
# Synthetic cases already carry -sub.nii.gz from augment_registration.py --with-subtraction.
python preprocess_replica.py \
  --config experiments/exp27_replica_5ch_regaug_5fold.yaml \
  --data_dir ../Brats2024/training_data1_v2 \
  --extra_data_dir ../Brats2024/training_data_additional \
  --num_processes 8
```

Two consequences of that asymmetry:

- **The section-4 layout is load-bearing for exp27.** If `../Brats2024/synthetic_regaug`
  does not resolve from the working directory, `collect_synthetic_case_dirs` raises
  `FileNotFoundError` (`splits.py:70`) — it fails loudly rather than silently training
  without the augmented data. Good, but it fails *after* you have paid for the pod.
- **Keep the real dirs relative too.** Passing absolute `--data_dir
  /workspace/Brats2024/...` while synthetic stays relative means the two halves can point
  at different trees. Use `../Brats2024/...` for both, as above.

Verify the synthetic cases actually landed:

Each case writes three files (`<case>.npy`, `<case>_seg.npy`, `<case>.pkl`), so count the
`.pkl` files to count cases:

```bash
C=/app/.cache/replica/Dataset102_5ch
ls $C/*.pkl | wc -l                  # ~3089 total (1621 real + 1468 synthetic at -k 1)
ls $C/*.pkl | grep -c -- -reg-       # ~1468 synthetic  (dir names carry -reg-)
```

Preprocessing is idempotent — re-running resumes and skips cached cases. Add `--fold 0` to
cache only what fold 0 needs if you want training to start sooner.

---

## 7. Experiment sequence

### What each experiment is

All nine run through the replica loop on the 11 GB ResEnc-M plan (patch [128,192,128],
batch 2). Each changes exactly one thing from exp20:

| exp | changes | read against |
|---|---|---|
| **20** | nothing — the baseline | the native run |
| **21** | +5th input channel (T1Gd−T1 subtraction) | exp20 |
| **22** | Dice-CE → Dice-Focal loss | exp20 |
| **23** | 5 channels **and** Dice-Focal | exp20/21/22 |
| **24** | ResEnc → PlainConvUNet plan (~102M → ~31M params; both planners chose the same patch and batch, so architecture is the only difference) | exp20 |
| **25** | network → MONAI SegResNetDS | exp20 |
| **26** | 1000 → 250 epochs | other 250-epoch runs only |
| **27** | + cross-patient registration augmentation (synthetic, train-side only) | **exp21** |
| **native** | the real nnU-Net v2 framework, same 11 GB plan | exp20 |

Caches split two ways: exp20/22/24/25/26 share `Dataset102_4ch`; exp21/23/27 share
`Dataset102_5ch`.

### Master sequence with decision gates

| # | step | budget | ~time | gate before continuing |
|---|---|---|---|---|
| 0 | Setup, data pull, 4ch cache (§4, §6) | — | ~2 h | — |
| 1 | Fit-check + 10 real epochs (§5) | — | ~30 min | **replace the s/epoch and GPU-util estimates with measurements** |
| 2 | **exp20 fold 0** | 1000 ep | ~13 h | — |
| 3 | **native nnU-Net fold 0** (§8) | 1000 ep | ~14 h | **does the replica reproduce native?** |
| 4 | exp20 folds 1–4 | 1000 ep | ~53 h | run only if parity holds |
| 5 | 5ch cache, then exp21/22/24/25/26/27 fold 0 | 250 ep | ~20 h | — |
| 6 | exp23 | 250 ep | ~3 h | **only if exp21 or exp22 showed an effect** |
| 7 | Top 1–2 from step 5, all folds | 1000 ep | ~134 h | — |
| 8 | OOF CV + held-out test + ensemble (§9) | — | ~4 h | — |

~240 GPU-hours total, ~10 days, ~$150.

**Why native runs at step 3 rather than last.** The parity check gates the whole series: if
the replica does not reproduce native nnU-Net, every exp21–27 delta is measured from a
suspect baseline. Learning that after ~27 h is far better than after paying for all five
exp20 folds. If the Windows box's native 11G run finishes first you get this for free — skip
step 3, or spend the slot on ResEnc-L as a separate, non-comparable datapoint.

Staged deliberately: nothing downstream is interpretable until the baseline lands, and
ranking questions are answered at 1/20th the cost of absolute ones.

All commands assume `cd /app`, the 11 GB variant, and this shared prefix:

```bash
R="python train_replica.py --data_dir ../Brats2024/training_data1_v2 \
   --extra_data_dir ../Brats2024/training_data_additional --no_grad_checkpointing"
```

### Tier 1 — exp20 baseline, 5 folds × 1000 epochs (~2.8 days, ~$40)

The parity anchor against the native run *and* the baseline every later delta is measured
from. Run this first, alone.

```bash
for f in 0 1 2 3 4; do
  $R --config experiments/exp20_replica_resenc_m_11g_5fold.yaml --fold $f
done
```

Resume any interruption in place by appending `--c` — it restores optimizer, GradScaler,
EMA and history intact.

### Tier 2 — screening, fold 0 × 250 epochs each (~20 h total, ~$12–15)

These are *ranking* questions. Six runs, each ~3.3 h.

```bash
for cfg in exp21_replica_5ch_subtraction_5fold \
           exp22_replica_dicefocal_5fold \
           exp24_replica_plain_unet_11g_5fold \
           exp25_replica_segresnet_ds_5fold \
           exp27_replica_5ch_regaug_5fold; do
  $R --config experiments/$cfg.yaml --fold 0 --epochs 250
done

# exp26 is already a 250-epoch config — run it as-is, no --epochs override
$R --config experiments/exp26_replica_short_budget_250ep_5fold.yaml --fold 0
```

Two rules that make or break the comparisons:

- **Compare only within a budget.** Tier-2 runs are 250-epoch and comparable to each other
  and to exp26 — never to Tier-1 exp20 at 1000. Poly LR is
  `0.01*(1-epoch/num_epochs)^0.9`, so a 250-epoch run has fully annealed while a
  1000-epoch run is still at 0.0077 at that point. The annealed model is usually better,
  for reasons that have nothing to do with the thing you are testing. **This is exactly
  why exp26 must be its own run and cannot be read off exp20's curve.**
- **exp27's baseline is exp21, not exp20.** exp21 is exp27 minus the synthetic data and
  their val/test sets are byte-identical. Reading exp27 against exp20 conflates the
  synthetic data with the 5th channel.

### Tier 3 — promote the winners, 5 folds × 1000 epochs (~$80)

Take the top 1–2 from Tier 2 and run them at full budget, same as Tier 1. Only pay full
price for what won.

### Conditional — exp23

exp23 is the 2×2 corner (5 channels **and** Dice-Focal). **Run it only if exp21 or exp22
showed an effect in Tier 2.** If both main effects are null, the interaction is not worth
the compute.

---

## 8. The native nnU-Net run

**Check the Windows box first.** A native ResEnc run on Dataset102 at the same
`-gpu_memory_target 11` plan is already scheduled on the 4070 SUPER (task `nnunet_11g`,
folds 0–4, log `nnunet_data\run_11g_*.log`). If that has finished, the parity anchor
exists and repeating it on the 4090 buys nothing.

That gives you two genuinely different options for the 4090's native slot:

**Option 1 — parity (recommended if the Windows run is unfinished or at risk).** Same
11 GB plan as the replica, so exp20 fold 0 and native fold 0 are directly comparable.
That comparison is the entire justification for the replica loop.

**First check which nnU-Net you actually have** — the planner class names differ by version
and the script hardcodes the newer ones:

```bash
python -c "import importlib.metadata as m; print(m.version('nnunetv2'))"
python -c "
from nnunetv2.experiment_planning.experiment_planners.resencUNet_planner import *
" 2>/dev/null && echo "has ResEnc planners"
```

- **≥ 2.4.2** (what `requirements.txt` pins, so what the container has): planners are
  `nnUNetPlannerResEncM` / `L` / `XL`. Use the commands below as written.
- **2.1** (what this laptop has — verified 2026-08-13, it exposes only
  `ExperimentPlanner` and `ResEncUNetPlanner`): substitute `-pl ResEncUNetPlanner`
  everywhere below. There the memory target is the only knob and there are no M/L/XL
  classes, so `run_resenc_5fold.sh --preset M` fails outright.

**`run_resenc_5fold.sh --preset M` will NOT give you parity either way.**
`nnUNetPlannerResEncM` targets **8 GB**, which on this dataset plans patch [128,160,112] —
not the replica's [128,192,128]. The script has no `-gpu_memory_target` flag
(`nnunet_native/run_resenc_5fold.sh:95`), so run the planning step by hand with an explicit
11 GB target, reusing the same plans name as the Windows box:

```bash
export nnUNet_raw=/app/nnunet_data/nnUNet_raw
export nnUNet_preprocessed=/app/nnunet_data/nnUNet_preprocessed
export nnUNet_results=/app/nnunet_data/nnUNet_results

# 1. Pool both real dirs into one tree. convert_to_nnunet.py takes a SINGLE --data_dir,
#    but the replica trains on the pooled 1621 cases — parity needs the same pool.
#    (The Windows box does this with directory junctions; symlinks are the Linux equivalent.)
mkdir -p ../Brats2024/training_all
ln -sfn ../training_data1_v2/*        ../Brats2024/training_all/
ln -sfn ../training_data_additional/* ../Brats2024/training_all/
ls ../Brats2024/training_all | wc -l          # expect 1621

# 2. Convert BraTS -> nnU-Net, holding out the seed-42 test patients into imagesTs
python nnunet_native/convert_to_nnunet.py \
  --data_dir ../Brats2024/training_all \
  --output_dir ./nnunet_data \
  --dataset_id 102 --dataset_name BraTS2024ResEnc \
  --mode kfold --n_folds 5 --split_seed 42

# 3. Plan at 11 GB. -overwrite_plans_name is REQUIRED whenever -gpu_memory_target
#    differs from the preset default, or nnU-Net overwrites the stock plans file.
nnUNetv2_plan_and_preprocess -d 102 \
  -pl nnUNetPlannerResEncM \
  -gpu_memory_target 11 \
  -overwrite_plans_name nnUNetResEncUNetPlans_11G \
  -c 3d_fullres --verify_dataset_integrity --verbose

# 4. Re-assert OUR patient-level folds — planning regenerates splits_final.json
#    with nnU-Net's own random KFold and would silently break comparability.
cp $nnUNet_raw/Dataset102_BraTS2024ResEnc/splits_final.json \
   $nnUNet_preprocessed/Dataset102_BraTS2024ResEnc/splits_final.json

# 5. Train fold 0
nnUNetv2_train 102 3d_fullres 0 -p nnUNetResEncUNetPlans_11G -tr nnUNetTrainer --npz
```

Confirm before training that `$nnUNet_preprocessed/Dataset102_BraTS2024ResEnc/nnUNetResEncUNetPlans_11G.json`
reports **patch_size [128,192,128], batch_size 2**. If it does not, the parity claim is void.

Then bridge the results into the shared metrics with steps 4–5 of the script
(`evaluate_nnunet_kfold.py` and `evaluate_nnunet.py`), passing
`-p nnUNetResEncUNetPlans_11G`.

**Option 2 — ResEnc-L at 24 GB.** `run_resenc_5fold.sh` defaults to `--preset L`, which is
designed for a 24 GB card and is the one thing the 4090 can do that neither the 3070 nor
the 4070 SUPER can. L gives patch [160,192,160], batch 3.

```bash
bash nnunet_native/run_resenc_5fold.sh \
  --data_dir ../Brats2024/training_data1_v2 \
  --preset L --folds 0 --gpu 0
```

**L is not comparable to exp20–27.** Different patch and batch size, so it answers "what
does a bigger plan buy on this dataset", not "does the replica match native". Report it as
its own line, never as a delta against the replica series.

Note from the local planner run: for *this* dataset M/L/XL are ~identical in parameter
count (102 M — topology caps at 6 stages and features hit the 320 ceiling); patch and batch
are what actually differ. Do not describe L as "a bigger model".

The script converts BraTS → nnU-Net format, holds out the same seed-42 test patients in
`imagesTs`, plans, preprocesses, and trains. Budget ~120 GB and several hours before the
first epoch. Start with `--folds 0`; add folds later only if fold 0 is informative.

---

## 9. Evaluation

```bash
# Out-of-fold CV across every completed fold
python evaluate_replica.py \
  --config experiments/exp20_replica_resenc_m_11g_5fold.yaml \
  --data_dir ../Brats2024/training_data1_v2 \
  --extra_data_dir ../Brats2024/training_data_additional \
  --run_dirs runs/replica_*_fold* --split val

# Leak-free held-out test, folds ensembled
python evaluate_replica.py ... --run_dirs runs/replica_*_fold* --split test
```

Post-processing has **two separate families** — nnU-Net's *determined* post-processing
(`--determine_postprocessing`, chosen on validation) and the BraTS heuristic
(`--postprocess`). Never report a number from one as if it came from the other, and never
determine post-processing on the test split.

Cross-model ensembling is `evaluate_ensemble.py`; averaging must happen in uncropped space.

---

## 10. Operations

**Spot instances are worth it here.** At ~48 s/epoch the `save_every: 50` checkpoint gap
costs ~40 minutes if preempted. Drop it to 10 for spot (`--epochs` unaffected; edit
`replica.save_every` in the YAML) — a 1.2 GB checkpoint every ~8 minutes is free on a
network volume. Always relaunch with `--c`.

**Run detached.** Every training command above should go through `nohup ... &` or `tmux`,
with output to a log on `/workspace`. An SSH drop must not kill a 13-hour fold.

**Monitor:**
```bash
tail -f /workspace/logs/exp20_fold0.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 30
tensorboard --logdir runs --host 0.0.0.0 --port 6006     # then use the pod's proxy URL
```

**Everything that matters lives on `/workspace`** — `runs/`, `.cache/`, `nnunet_data/`,
logs. Container disk is wiped when the pod is terminated; the network volume is not.

**Before terminating a pod:** confirm checkpoints are on the volume, not container disk.
`shred -u` the GCS service-account key. Keep the volume if another box is coming; delete
it when the series is done, along with the GCS bucket
(`gcloud storage rm -r $BUCKET/tar`).

---

## 11. Cost summary

At the plan's assumed ~$0.60/h for a community-cloud 4090:

| stage | GPU time | cost |
|---|---|---|
| Tier 1 — exp20, 5 folds × 1000 ep | ~67 h | ~$40 |
| Tier 2 — 6 screening runs, fold 0 × 250 ep | ~20 h | ~$12–15 |
| Tier 3 — 1–2 promotions, 5 folds × 1000 ep | ~134 h | ~$80 |
| Native nnU-Net, fold 0 | ~14 h | ~$9 |
| Network volume 500 GB | continuous | check current RunPod rate |
| GCS egress, one full pull | — | ~$10.50 |

**~$150 total.** Every row scales off the estimated 3× speedup over the 3070 and the
repo's own ~2× no-checkpointing claim — neither measured on cloud hardware. Section 5's
ten-epoch check replaces both estimates with a real number for about a dollar. Do it
before booking Tier 1.
