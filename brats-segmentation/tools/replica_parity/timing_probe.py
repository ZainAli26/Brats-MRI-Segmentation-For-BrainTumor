#!/usr/bin/env python3
"""Where does the epoch-time gap come from? Time the GPU step in three configurations.

The replica's measured epoch is ~2.2x the native steady-state epoch even though both sit at
100% GPU utilisation. The only difference in the compute path is that nnU-Net 2.4.2 defaults
`nnUNet_compile` to ON and the replica never calls torch.compile. This times, on identical
fixed batches, run sequentially so each has the whole card:

  1. native trainer, torch.compile ON   (nnU-Net as shipped)
  2. native trainer, torch.compile OFF  (the same code, compile removed)
  3. replica trainer                    (never compiles)

    PYTHONPATH=<nnUNet242> python3 timing_probe.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

# Durable work dir. This lived in the session scratchpad once; a reboot destroyed it
# along with the cache and every run. nnunet_data/ is gitignored, so heavy artefacts
# stay out of git while these scripts stay in it.
SP = Path("/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/"
          "brats-segmentation/nnunet_data/replica_parity")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from equiv_checks2 import fixed_batch, native_trainer, replica_trainer  # noqa: E402

WARMUP, MEASURE = 6, 15


def time_steps(trainer, batches, label):
    net = trainer.network
    net.train()
    trainer.optimizer = torch.optim.SGD(net.parameters(), 1e-2, weight_decay=3e-5,
                                        momentum=0.99, nesterov=True)
    trainer.grad_scaler = torch.amp.GradScaler("cuda")
    for i in range(WARMUP):
        trainer.train_step({"data": batches[i % len(batches)]["data"].clone(),
                            "target": [t.clone() for t in batches[i % len(batches)]["target"]]})
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(MEASURE):
        trainer.train_step({"data": batches[i % len(batches)]["data"].clone(),
                            "target": [t.clone() for t in batches[i % len(batches)]["target"]]})
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / MEASURE
    peak = torch.cuda.max_memory_reserved() / 2 ** 30
    print(f"{label:<34}{dt:>8.3f} s/step   {dt * 250:>8.1f} s / 250-iter epoch   "
          f"peak reserved {peak:.2f} GiB", flush=True)
    return dt, peak


def main():
    import os
    results = {}

    os.environ["nnUNet_compile"] = "1"
    nat = native_trainer()
    patch = list(nat.configuration_manager.patch_size)
    scales = nat._get_deep_supervision_scales()
    batches = [fixed_batch(patch, scales, nat.batch_size, seed=300 + i, train=True) for i in range(3)]
    torch.cuda.reset_peak_memory_stats()
    results["native_compile_on"] = time_steps(nat, batches, "native, torch.compile ON")
    del nat
    torch.cuda.empty_cache()

    os.environ["nnUNet_compile"] = "0"
    nat2 = native_trainer()
    torch.cuda.reset_peak_memory_stats()
    results["native_compile_off"] = time_steps(nat2, batches, "native, torch.compile OFF")
    del nat2
    torch.cuda.empty_cache()

    rep = replica_trainer()
    torch.cuda.reset_peak_memory_stats()
    results["replica"] = time_steps(rep, batches, "replica (no compile)")
    del rep
    torch.cuda.empty_cache()

    a = results["native_compile_on"][0]
    b = results["native_compile_off"][0]
    c = results["replica"][0]
    print(f"\ntorch.compile speedup inside nnU-Net: {b / a:.2f}x")
    print(f"replica vs native-with-compile:       {c / a:.2f}x slower")
    print(f"replica vs native-without-compile:    {c / b:.2f}x  <- 1.0 means the loops are equally fast")
    json.dump({k: {"s_per_step": v[0], "peak_gib": v[1]} for k, v in results.items()},
              open(SP / "timing_probe.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())

