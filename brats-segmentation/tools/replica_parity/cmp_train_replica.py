#!/usr/bin/env python3
"""Train the replica loop on Dataset500_ReplicaCmp fold 0 — the mirror of the native run.

Same plan, same 201 cases, same fold-0 partition, same epoch budget. Uses
NNUNetReplicaTrainer directly (rather than train_replica.py) so the case lists are the
native run's splits_final.json fold 0 verbatim instead of being re-derived from the raw
BraTS directories.

    python3 cmp_train_replica.py --epochs 2 [--grad_checkpointing] [--tag name]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path("/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/brats-segmentation")
# Durable work dir. This lived in the session scratchpad once; a reboot destroyed it
# along with the cache and every run. nnunet_data/ is gitignored, so heavy artefacts
# stay out of git while these scripts stay in it.
SP = Path("/home/zain/workspace/repos/Brats-MRI-Segmentation-For-BrainTumor/"
          "brats-segmentation/nnunet_data/replica_parity")
sys.path.insert(0, str(REPO))

torch.backends.cudnn.benchmark = True

from src.nnunet_replica.config import ReplicaConfig          # noqa: E402
from src.nnunet_replica.trainer import NNUNetReplicaTrainer  # noqa: E402
from src.utils.experiment import load_config                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--val_iters", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--grad_checkpointing", action="store_true")
    ap.add_argument("--tag", default="replica_cmp")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override replica.seed — used to measure run-to-run spread")
    ap.add_argument("--final_validation", action="store_true",
                    help="Run the full-image sliding-window validation pass afterwards")
    args = ap.parse_args()

    # Start from exp20's config so class names / eval regions / label handling are the ones
    # the experiment series uses; retarget it at the 8 GB plan and the comparison cache.
    config = load_config(str(REPO / "experiments/exp20_replica_resenc_m_11g_5fold.yaml"))
    config["replica"].update({
        "plans_json": str(REPO / "plans/nnUNetResEncUNetMPlans_8G_cmp.json"),
        "preprocessed_dir": str(SP / "replica_cache/Dataset500_ReplicaCmp_fp32"),
        "store_dtype": "float32",
        "num_epochs": args.epochs,
        "num_iterations_per_epoch": args.iters,
        "num_val_iterations_per_epoch": args.val_iters,
        "num_workers": args.num_workers,
        "grad_checkpointing": bool(args.grad_checkpointing),
        "validate_with_full_images": bool(args.final_validation),
        "save_every": 1,
    })
    config["experiment"]["output_dir"] = str(SP / "runs")

    if args.seed is not None:
        config["replica"]["seed"] = args.seed

    rcfg = ReplicaConfig.from_config(config)

    # The native run's fold 0, verbatim.
    split = json.load(open(REPO / "nnunet_data/nnUNet_preprocessed/Dataset500_ReplicaCmp"
                           / "splits_final.json"))[0]
    train_ids, val_ids = list(split["train"]), list(split["val"])

    out = SP / "runs" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"replica run -> {out}\n  train {len(train_ids)}  val {len(val_ids)}  "
          f"epochs {rcfg.num_epochs} x {rcfg.num_iterations_per_epoch}  "
          f"grad_checkpointing={rcfg.grad_checkpointing}", flush=True)

    trainer = NNUNetReplicaTrainer(
        config=config, replica_cfg=rcfg, fold=0,
        train_case_ids=train_ids, val_case_ids=val_ids,
        output_folder=out, tracker=None,
    )
    best_ema = trainer.run_training()
    print(f"best EMA pseudo Dice: {best_ema}")
    print(f"peak GPU alloc: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")

    summary = {"best_ema_pseudo_dice": best_ema,
               "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
               "logging": trainer.logger.logging}
    if args.final_validation:
        summary["full_image_validation"] = trainer.perform_actual_validation()
    json.dump(summary, open(out / "cmp_summary.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())

