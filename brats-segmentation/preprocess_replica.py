#!/usr/bin/env python3
"""Build the preprocessed case cache the replica trainer reads.

Runs nnU-Net's preprocessing (crop to brain → masked z-score → resample to the plan's
spacing → sample foreground locations) once per case and writes it in nnU-Net's unpacked
layout so training memory-maps patches instead of decompressing four .nii.gz per sample.

    python preprocess_replica.py --config experiments/exp20_replica_resenc_m_11g_5fold.yaml
    python preprocess_replica.py --config <cfg> --num_processes 8 --max_cases 20

Re-running skips cases already present, so an interrupted pass just resumes. Expect
~30 GB (fp16) for the 1621-case BraTS 2024 pool; pass --store_dtype float32 to match
nnU-Net byte for byte at double the size.
"""

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.data.dataset import build_file_list
from src.nnunet_replica.plans import Plans
from src.nnunet_replica.preprocessing import (
    case_is_cached,
    preprocess_case,
    save_preprocessed_case,
)
from src.nnunet_replica.config import ReplicaConfig
from src.nnunet_replica.splits import build_replica_splits
from src.utils.experiment import load_config

console = Console()


def _run_one(entry, plans_json, configuration, out_dir, foreground_labels, label_map,
             store_dtype, transpose_forward):
    case_id = entry["case_id"]
    try:
        cfg = Plans.load(plans_json).get_configuration(configuration)
        image_files = [v for k, v in entry.items() if k.startswith("image_")]
        data, seg, props = preprocess_case(
            image_files, entry.get("label"), cfg, foreground_labels,
            label_map=label_map, transpose_forward=transpose_forward,
        )
        save_preprocessed_case(Path(out_dir), case_id, data, seg, props, store_dtype)
        return case_id, None
    except Exception:
        return case_id, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser(description="Preprocess BraTS cases for the nnU-Net replica loop")
    ap.add_argument("--config", required=True, help="Experiment YAML with a `replica:` block")
    ap.add_argument("--data_dir", help="Override data.train_dir")
    ap.add_argument("--extra_data_dir", action="append", default=None,
                    help="Override data.extra_train_dirs (repeatable)")
    ap.add_argument("--output_dir", help="Override replica.preprocessed_dir")
    ap.add_argument("--num_processes", type=int, default=6)
    ap.add_argument("--fold", type=int, default=None,
                    help="Preprocess only the cases this fold needs, so fold N can start "
                         "before the whole cache is built")
    ap.add_argument("--max_cases", type=int, default=None, help="Preprocess only the first N (smoke test)")
    ap.add_argument("--store_dtype", default=None, choices=["float16", "float32"])
    ap.add_argument("--overwrite", action="store_true", help="Re-preprocess cases already cached")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["data"]["train_dir"] = args.data_dir
    if args.extra_data_dir:
        config["data"]["extra_train_dirs"] = args.extra_data_dir

    rcfg = ReplicaConfig.from_config(config)
    if args.output_dir:
        rcfg.preprocessed_dir = args.output_dir
    store_dtype = args.store_dtype or config.get("replica", {}).get("store_dtype", "float16")

    plans = Plans.load(rcfg.plans_json)
    console.print(f"[bold cyan]{plans.summary(rcfg.configuration)}[/bold cyan]")

    folds, _, all_case_dirs = build_replica_splits(config)
    if args.fold is not None:
        if not 0 <= args.fold < len(folds):
            console.print(f"[red]fold {args.fold} out of range (0..{len(folds) - 1})[/red]")
            sys.exit(1)
        wanted = set(folds[args.fold][0]) | set(folds[args.fold][1])
        all_case_dirs = [d for d in all_case_dirs if d.name in wanted]
        console.print(f"[dim]fold {args.fold}: {len(all_case_dirs)} cases[/dim]")
    if args.max_cases:
        all_case_dirs = all_case_dirs[: args.max_cases]

    modalities = config["data"]["modalities"]
    entries = build_file_list(all_case_dirs, modalities, include_label=True)
    if not entries:
        console.print("[red]No cases found — check data.train_dir / modalities[/red]")
        sys.exit(1)

    out_dir = rcfg.resolved_preprocessed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        todo = [e for e in entries if not case_is_cached(out_dir, e["case_id"])]
    else:
        todo = entries

    console.print(f"[bold]{len(entries)} cases total, {len(todo)} to preprocess "
                  f"-> {out_dir} ({store_dtype})[/bold]")
    if not todo:
        console.print("[green]Cache already complete.[/green]")
        return

    label_map = {int(k): int(v) for k, v in config["data"]["label_map"].items()}
    foreground_labels = list(range(1, int(config["data"]["num_classes"])))
    failures = []

    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                  TimeRemainingColumn(), console=console) as progress:
        task = progress.add_task("preprocessing", total=len(todo))
        with ProcessPoolExecutor(max_workers=args.num_processes) as pool:
            futures = [
                pool.submit(_run_one, e, rcfg.plans_json, rcfg.configuration, str(out_dir),
                            foreground_labels, label_map, store_dtype, plans.transpose_forward)
                for e in todo
            ]
            for fut in as_completed(futures):
                case_id, err = fut.result()
                if err:
                    failures.append((case_id, err))
                    console.print(f"[red]FAILED {case_id}[/red]")
                progress.update(task, advance=1)

    if failures:
        console.print(f"[red bold]{len(failures)} case(s) failed:[/red bold]")
        for case_id, err in failures[:3]:
            console.print(f"[red]{case_id}[/red]\n{err}")
        sys.exit(1)
    console.print(f"[bold green]Done. Cache at {out_dir}[/bold green]")


if __name__ == "__main__":
    main()
