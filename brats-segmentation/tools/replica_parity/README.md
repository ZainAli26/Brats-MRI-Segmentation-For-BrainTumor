# Replica ⇄ native parity harness

Evidence that `src/nnunet_replica/` reproduces nnU-Net v2, rather than merely resembling it.

The question this answers is narrow and worth stating precisely: **does our loop compute the
same thing native nnU-Net computes?** Two 2-epoch runs cannot answer that on their own —
they differ by RNG stream and data order, so any gap between them is confounded. The real
answer comes from feeding *identical weights and identical batches* to both and comparing
the outputs. The end-to-end runs are the sanity check on top.

## Why this directory exists

The first version of this harness lived in a session scratchpad under `/tmp`. A power loss
rebooted the machine and destroyed all of it — the nnU-Net 2.4.2 checkout, the replica
cache, every script, and the replica run. The numbers survived only because they had been
printed into a transcript. Everything here is now in git; heavy artefacts go to
`nnunet_data/replica_parity/`, which is gitignored.

## What is compared

Dataset500_ReplicaCmp — a patient-grouped subset of exp20's fold 0, 161 train / 40 val
cases, built from the same BraTS directories the experiment series uses. Both sides run
`nnUNetResEncUNetMPlans` (the genuine 8 GB ResEnc-M planner output): patch
`[128, 160, 112]`, batch 2, `batch_dice: False`, last stride `(2, 2, 1)`, 101,945,177
parameters.

The native side is nnU-Net **2.4.2** on `PYTHONPATH` rather than installed, so torch,
batchgenerators and dynamic_network_architectures stay byte-identical to the ones the
replica imports. A version difference in a shared library would invalidate the comparison.

Two patches are applied to that checkout, both compatibility-only:

- `polylr.py` — torch ≥ 2.7 removed `LRScheduler`'s positional `verbose`; 2.4.2 still passes it.
- `nnUNetTrainer_2epochs.py` — added following nnU-Net's own `nnUNetTrainer_Xepochs` pattern.

`equiv_checks2.py` additionally shims `torch.load` back to `weights_only=False`, which
torch ≥ 2.6 flipped. None of the three change what is computed.

## Scripts

| script | what it does |
|---|---|
| `build_cmp_dataset.py` | builds Dataset500 (symlinks + `dataset.json`) from exp20's fold-0 split |
| `nnu.sh` | runs a 2.4.2 entry point (`fingerprint`/`plan`/`preprocess`/`train`/`predict`) against it |
| `cmp_preprocess_replica.py` | builds the replica's cache from the *same files*, `float32` so it can match nnU-Net byte for byte |
| `cmp_train_replica.py` | replica training on the native run's `splits_final.json` fold 0 verbatim |
| `equiv_checks.py` | 9 component checks: plan, DS scales/weights, network+init, loss, LR, rotation, augmentation, sampler, preprocessing |
| `equiv_checks2.py` | the decisive ones: `train_step`, `validation_step`, and full inference, all on identical weights + batches |
| `compare_runs.py` | head-to-head table of the two completed runs |
| `timing_probe.py` | isolates `torch.compile` as the source of the epoch-time gap |

## Running it

```bash
B=brats-segmentation; W=$B/nnunet_data/replica_parity

# One-time: nnU-Net 2.4.2 + the two compat patches (see above), then the replica cache.
git clone --depth 1 --branch v2.4.2 https://github.com/MIC-DKFZ/nnUNet.git $W/nnUNet242
python3 tools/replica_parity/cmp_preprocess_replica.py          # ~2 min, ~11 GB

# Component equivalence — no training needed.
PYTHONPATH=$W/nnUNet242 nnUNet_raw=$B/nnunet_data/nnUNet_raw \
  nnUNet_preprocessed=$B/nnunet_data/nnUNet_preprocessed \
  nnUNet_results=$B/nnunet_data/nnUNet_results nnUNet_compile=0 \
  python3 tools/replica_parity/equiv_checks.py

# Step-level equivalence. nnUNet_compile=0 matters: torch.compile perturbs the
# arithmetic enough to blur a 1e-6 comparison, and it is also the epoch-time gap.
... python3 tools/replica_parity/equiv_checks2.py train_step val_step inference

# End-to-end, one side at a time — two 102 M-param nets do not share an 8 GB card.
tools/replica_parity/nnu.sh train 500 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_2epochs
python3 tools/replica_parity/cmp_train_replica.py --epochs 2 --tag replica_cmp
python3 tools/replica_parity/compare_runs.py replica_cmp
```

`cmp_train_replica.py --seed N` overrides `replica.seed`, which is how run-to-run spread is
measured — without that number, a difference between the two end-to-end runs cannot be
told apart from noise.

Results are written to `NNUNET_REPLICA.md` (§ Parity with native nnU-Net).
