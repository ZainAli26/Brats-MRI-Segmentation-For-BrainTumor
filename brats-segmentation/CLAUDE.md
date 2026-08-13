# BraTS segmentation — Claude Code operating instructions

This directory is the working root for the exp20–27 replica series and the native nnU-Net
runs. Every command runs from here.

**Read `RUNPOD_4090_GUIDE.md` before doing anything on a fresh box.** It holds the actual
commands, sizes, and gotchas. This file holds the *rules* — what you may do unattended and
what you must hand back to the user.

---

## THE HARD RULE — never start a training run

Training runs cost 13–134 hours of rented GPU each. Starting one on a wrong config wastes
days and real money, and the decision to spend is the user's.

**When the procedure reaches a training step: do not run it. Print the exact command and
stop.** Use this format:

> **Ready to train: exp20 fold 0.** Run this yourself:
> ```bash
> cd /app && nohup python train_replica.py \
>   --config experiments/exp20_replica_resenc_m_11g_5fold.yaml \
>   --data_dir ../Brats2024/training_data1_v2 \
>   --extra_data_dir ../Brats2024/training_data_additional \
>   --fold 0 --no_grad_checkpointing --num_workers 22 \
>   > /workspace/logs/exp20_fold0.log 2>&1 &
> ```
> Expected ~13 h. Log: `/workspace/logs/exp20_fold0.log`.
> Resume after any interruption: same command with `--c` appended.
> Watch for: s/epoch in the log, and `nvidia-smi` utilization above ~60%.

Then wait. Do not poll for hours; do not launch it "to save time".

### Requires the user to run it

- `train_replica.py` **without** `--smoke_test`
- `nnUNetv2_train`, or `run_resenc_5fold.sh` (it trains internally)
- the 10-epoch benchmark in guide §5 — it is short but it is still a real training loop
- anything you expect to occupy the GPU for more than ~15 minutes

### You may run unattended

- environment audit, `nvidia-smi`, `nproc`, `df`, `free`, `fio`
- installs (`pip install`, gcloud SDK)
- the GCS data pull and extraction, and its verification counts
- symlink/layout setup
- `preprocess_replica.py`, `precompute_subtraction.py` (CPU-only, idempotent, ~25 min)
- `train_replica.py --smoke_test` and the fit-check (guide §5 step 1, ~2 min)
- `nnUNetv2_plan_and_preprocess`, `convert_to_nnunet.py` (CPU-only)
- `evaluate_replica.py`, `evaluate_kfold.py`, `evaluate_ensemble.py` on **finished** runs
- reading logs, diagnosing failures, computing metrics

### Never without explicit approval

- deleting or overwriting `Brats2024/`, `.cache/`, `runs/`, `nnunet_data/`
- `git commit` or `git push` (this repo is **public** — see below)
- `gcloud storage rm`
- editing any `experiments/*.yaml` — the configs define the experiments; a silent edit
  invalidates comparisons. Propose the diff and wait.

---

## Bootstrap sequence on a fresh box

Work through these in order. **End each phase with a short report and the verification
output**, then continue to the next phase unless something failed. Record progress in
`/workspace/RUNPOD_STATE.md` (create it if absent) so a later session can resume without
redoing work.

| phase | what | guide § | done when |
|---|---|---|---|
| 0 | Audit: GPU, `nproc`, RAM, `df -h /workspace /`, `df -h /dev/shm` | §3 | numbers reported; `/dev/shm` ≥ 8 G or remounted |
| 1 | Install deps (strip `torch` and `antspyx` from requirements) | §2 Path B | `torch.cuda.is_available()` is True on a `+cu12x` build |
| 2 | Confirm GCS auth, then pull + extract tarballs detached | §4 | counts are 8100 / 1626 / 8808 |
| 3 | Layout: `/Brats2024`, `/app/.cache` on container disk, `runs` + `nnunet_data` on volume | §4 | `cd /app && ls ../Brats2024 && readlink -f .cache runs` looks right |
| 4 | Verify `-sub` channels, then build the 4ch cache | §6 | 1350 / 271 / 1468 sub files; cache case count matches |
| 5 | Fit-check without gradient checkpointing | §5 | peak VRAM ~11 GB via `nvidia-smi`, no OOM |
| 6 | **Hand off exp20 fold 0** | §7 | user has the command |
| 7 | After each run finishes: evaluate, report metrics | §9 | — |

Set `--num_workers` from the box's actual `nproc` minus 2, not from the config's 12 — see
guide §1 for why the 4090 needs far more workers than the 3070 did.

**Interactive auth is the user's step.** You cannot complete a browser login. Check
`gcloud auth list` first; if no account is active, ask the user to run it and tell them they
can prefix it with `!` in the Claude Code prompt so the output lands in the session:

> `! gcloud auth activate-service-account --key-file=/root/key.json`

Then continue. The same applies to anything else needing a browser or a password.

---

## Facts that change decisions — do not rediscover these

- **`docker-compose.yml` is a reference, not a runner.** A RunPod pod is itself a container;
  use plain `python` commands.
- **The 11 GB variant means `--no_grad_checkpointing`.** It is the default for this series.
- **exp27's synthetic dir cannot be set on the CLI.** `data.synthetic_train_dirs` is read
  from the YAML only, so the §4 layout must make `../Brats2024/synthetic_regaug` resolve.
- **The `-sub` 5th channel is already precomputed** for all 1621 real and 1468 synthetic
  cases, and ships inside the tarballs. Verify; do not regenerate by default.
- **Two caches:** `Dataset102_4ch` (exp20/22/24/25/26) and `Dataset102_5ch` (exp21/23/27).
- **`--preset M` does not give native/replica parity** — it targets 8 GB. Parity needs
  `-gpu_memory_target 11 -overwrite_plans_name nnUNetResEncUNetPlans_11G`, and the native
  side must train on the pooled 1621 cases. Guide §8.
- **Checkpoints are not written atomically** (`trainer.py:422` is a bare `torch.save`), so
  never rsync a checkpoint that may be mid-write, and expect a truncated
  `checkpoint_latest.pth` if a pod dies during a save.
- **The loop is CPU-bound on augmentation**, ~11 s per batch single-threaded. If GPU
  utilization is low, raise `--num_workers` before suspecting anything else.

## Comparison rules — a violated one silently invalidates a result

- **exp27 is read against exp21**, never exp20.
- **Only compare runs at the same epoch budget.** 250-epoch runs compare to each other and
  to exp26; never to a 1000-epoch run. Poly LR annealing, not the treatment, dominates.
- **exp23 only runs if exp21 or exp22 showed an effect.**
- **Never determine post-processing on the test split**, and never report nnU-Net
  *determined* post-processing as if it were the BraTS heuristic. They are separate
  families.

## Security

- This repo is **public**. Never commit bucket names, GCP project ids, emails, or keys.
  Use `<your-bucket>` / `<your-project>` placeholders in any doc you write.
- The GCS service-account key is a bearer credential on a rented machine. Keep it at
  `/root/key.json`, never inside the repo, and `shred -u` it when the pull is done.
- If the user asks you to commit, check `git status` for stray data or credentials first.

## Reporting

Report measurements, not reassurance. Give the number and where it came from
(`nvidia-smi`, the log line, the file count). If a verification count is off, say so and
stop rather than proceeding on the assumption it is close enough. If you skipped a step,
say which and why.
