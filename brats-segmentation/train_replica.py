#!/usr/bin/env python3
"""Train one fold with the nnU-Net replica loop.

    python train_replica.py --config experiments/exp20_replica_resenc_m_11g_5fold.yaml --fold 0
    python train_replica.py --config <cfg> --fold 0 --c            # resume
    python train_replica.py --config <cfg> --fold 0 --smoke_test   # 2 epochs x 5 iters

Requires the preprocessed cache (see preprocess_replica.py). Writes a run directory under
``experiment.output_dir`` containing nnU-Net-style artefacts (training_log_*.txt,
progress.png, checkpoint_best/latest/final.pth, validation_summary.json) plus the repo's
usual config.yaml / TensorBoard logs.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel

torch.backends.cudnn.benchmark = True

from src.nnunet_replica.config import ReplicaConfig
from src.nnunet_replica.splits import build_replica_splits, save_splits
from src.nnunet_replica.trainer import NNUNetReplicaTrainer
from src.utils.experiment import ExperimentTracker, load_config

console = Console()


def main():
    ap = argparse.ArgumentParser(description="nnU-Net replica training (custom loop)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--data_dir")
    ap.add_argument("--extra_data_dir", action="append", default=None)
    ap.add_argument("--preprocessed_dir", help="Override replica.preprocessed_dir")
    ap.add_argument("--output_dir", help="Override experiment.output_dir")
    ap.add_argument("--c", "--continue", dest="resume", action="store_true",
                    help="Resume this fold from checkpoint_latest.pth")
    ap.add_argument("--resume_dir", help="Explicit run directory to resume into")
    ap.add_argument("--epochs", type=int, help="Override replica.num_epochs")
    ap.add_argument("--iters_per_epoch", type=int, help="Override replica.num_iterations_per_epoch")
    ap.add_argument("--val_iters", type=int, help="Override replica.num_val_iterations_per_epoch")
    ap.add_argument("--num_workers", type=int, help="Override replica.num_workers")
    # Hardware-fitting overrides. The plan is fixed; these only change how it is made to
    # fit the GPU in front of you, so the same config runs on the 8 GB laptop and the
    # 12 GB desktop without editing YAML.
    ap.add_argument("--grad_checkpointing", dest="grad_ckpt", action="store_true", default=None,
                    help="Force encoder gradient checkpointing on (needed below ~11 GB VRAM)")
    ap.add_argument("--no_grad_checkpointing", dest="grad_ckpt", action="store_false",
                    help="Force it off — ~2x faster steps, needs the plan's full 11 GB budget")
    ap.add_argument("--batch_size", type=int,
                    help="Override the plan's batch size (deviates from the replica)")
    ap.add_argument("--grad_accum_steps", type=int,
                    help="Split the batch into N micro-steps. Exact while batch_dice is False")
    ap.add_argument("--smoke_test", action="store_true",
                    help="2 epochs x 5 train / 2 val iterations — validates wiring, not accuracy")
    ap.add_argument("--max_cases", type=int, default=None,
                    help="Use only the first N train and N val cases (for smoke tests on a "
                         "partially preprocessed cache)")
    ap.add_argument("--no_final_validation", action="store_true",
                    help="Skip the full-image sliding-window validation pass at the end")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["data"]["train_dir"] = args.data_dir
    if args.extra_data_dir:
        config["data"]["extra_train_dirs"] = args.extra_data_dir
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir

    rcfg = ReplicaConfig.from_config(config)
    if args.preprocessed_dir:
        rcfg.preprocessed_dir = args.preprocessed_dir
    if args.epochs:
        rcfg.num_epochs = args.epochs
    if args.iters_per_epoch:
        rcfg.num_iterations_per_epoch = args.iters_per_epoch
    if args.val_iters:
        rcfg.num_val_iterations_per_epoch = args.val_iters
    if args.num_workers is not None:
        rcfg.num_workers = args.num_workers
    if args.grad_ckpt is not None:
        rcfg.grad_checkpointing = args.grad_ckpt
    if args.batch_size:
        rcfg.batch_size = args.batch_size
    if args.grad_accum_steps:
        rcfg.grad_accum_steps = args.grad_accum_steps
    rcfg.validate()
    if args.smoke_test:
        rcfg.num_epochs = 2
        rcfg.num_iterations_per_epoch = 5
        rcfg.num_val_iterations_per_epoch = 2
        rcfg.save_every = 1

    if not rcfg.resolved_preprocessed_dir().is_dir():
        console.print(f"[red bold]Preprocessed cache not found: {rcfg.resolved_preprocessed_dir()}\n"
                      f"Run: python preprocess_replica.py --config {args.config}[/red bold]")
        sys.exit(1)

    folds, test_ids, _ = build_replica_splits(config)
    if not 0 <= args.fold < len(folds):
        console.print(f"[red]fold {args.fold} out of range (0..{len(folds) - 1})[/red]")
        sys.exit(1)
    train_ids, val_ids = folds[args.fold]
    if args.max_cases:
        train_ids, val_ids = train_ids[: args.max_cases], val_ids[: args.max_cases]
        console.print(f"[yellow]--max_cases {args.max_cases}: this is a wiring check, "
                      f"not a valid result[/yellow]")

    console.print(Panel.fit(
        f"[bold cyan]nnU-Net replica — fold {args.fold}[/bold cyan]\n"
        f"[dim]plans: {rcfg.plans_json} / {rcfg.configuration}\n"
        f"train {len(train_ids)} cases | val {len(val_ids)} | held-out test {len(test_ids)}\n"
        f"{rcfg.num_epochs} epochs x {rcfg.num_iterations_per_epoch} iters[/dim]",
        border_style="bright_blue",
    ))

    # Run directory: <output_dir>/<name>_fold<N>, reused verbatim when resuming.
    output_dir = Path(config["experiment"]["output_dir"]).expanduser()
    if args.resume_dir:
        run_dir = Path(args.resume_dir)
    elif args.resume:
        candidates = sorted(output_dir.glob(f"replica_*_fold{args.fold}"))
        if not candidates:
            console.print(f"[red]--c given but no replica_*_fold{args.fold} run in {output_dir}[/red]")
            sys.exit(1)
        run_dir = candidates[-1]
    else:
        run_dir = None

    tracker = ExperimentTracker(config, config_path=args.config,
                               resume_dir=str(run_dir) if run_dir else None)
    if run_dir is None:
        # ExperimentTracker names runs "<model>_<timestamp>"; tag it as a replica fold.
        new_dir = tracker.run_dir.parent / f"replica_{tracker.run_dir.name}_fold{args.fold}"
        tracker.run_dir.rename(new_dir)
        tracker.run_dir = new_dir
        tracker.logs_dir = new_dir / "logs"
        tracker.checkpoints_dir = new_dir / "checkpoints"
        tracker.viz_dir = new_dir / "visualizations"
        for d in (tracker.logs_dir, tracker.checkpoints_dir, tracker.viz_dir):
            d.mkdir(exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        tracker.writer = SummaryWriter(log_dir=str(tracker.logs_dir))
        tracker.log_file = new_dir / "training.log"
        run_dir = new_dir

    save_splits(folds, test_ids, run_dir / "splits.json")

    trainer = NNUNetReplicaTrainer(
        config=config, replica_cfg=rcfg, fold=args.fold,
        train_case_ids=train_ids, val_case_ids=val_ids,
        output_folder=run_dir, tracker=tracker,
    )

    latest = run_dir / "checkpoint_latest.pth"
    if (args.resume or args.resume_dir) and latest.is_file():
        trainer.load_checkpoint(latest)
    elif args.resume or args.resume_dir:
        console.print(f"[yellow]No checkpoint_latest.pth in {run_dir} — starting fresh[/yellow]")

    best_ema = trainer.run_training()

    summary = {"fold": args.fold, "best_ema_pseudo_dice": best_ema,
               "train_cases": len(train_ids), "val_cases": len(val_ids),
               "plans": rcfg.plans_json, "configuration": rcfg.configuration}

    if rcfg.validate_with_full_images and not args.no_final_validation:
        console.print("\n[bold]Full-image validation (sliding window + mirroring TTA)...[/bold]")
        summary["full_image_validation"] = trainer.perform_actual_validation()

    tracker.save_summary(summary)
    tracker.close()
    console.print(f"\n[bold green]Run saved to {run_dir}[/bold green]")
    console.print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
