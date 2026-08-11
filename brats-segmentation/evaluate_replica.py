#!/usr/bin/env python3
"""Evaluate replica runs: out-of-fold CV Dice/HD95, or a 5-fold ensemble on the test set.

    # per-fold validation (the CV metric) — pass every fold to pool them into one OOF number
    python evaluate_replica.py --config experiments/exp20_replica_resenc_m_11g_5fold.yaml \
        --run_dirs runs/replica_*_fold0 runs/replica_*_fold1 ... --split val

    # leak-free held-out test, 5-fold ensemble (softmax averaged across folds)
    python evaluate_replica.py --config <cfg> --run_dirs runs/replica_*_fold[0-4] --split test

``--split val`` scores each fold's own validation cases with that fold's network, which is
what "out-of-fold" means and what the native `nnUNetv2_train` validation reports.
``--split test`` scores the seed-42 10 % held-out patients — never seen by any fold — with
all folds ensembled, so it is directly comparable to the native ensemble prediction.

Inference is the replica's own (Gaussian tiling + mirroring TTA); ``inference.postprocess``
in the config additionally applies the repo's connected-component cleanup, letting you
report with and without it from the same checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.nnunet_replica.config import ReplicaConfig
from src.nnunet_replica.dataloading import PreprocessedDataset
from src.nnunet_replica.inference import predict_sliding_window
from src.nnunet_replica.network import build_network, set_deep_supervision_enabled
from src.nnunet_replica.plans import Plans
from src.nnunet_replica.splits import build_replica_splits
from src.utils.experiment import load_config

console = Console()


def load_fold_network(run_dir: Path, checkpoint: str, plan_cfg, num_input_channels: int,
                      num_classes: int, device: torch.device) -> torch.nn.Module:
    ckpt_path = run_dir / f"checkpoint_{checkpoint}.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"{ckpt_path} not found")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    net = build_network(plan_cfg, num_input_channels, num_classes,
                        deep_supervision=True, grad_checkpointing=False)
    net.load_state_dict(ckpt["network_weights"])
    set_deep_supervision_enabled(net, False)   # inference uses the full-resolution head only
    return net.to(device).eval()


def hd95(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """95th-percentile Hausdorff distance in mm; NaN when either mask is empty."""
    from monai.metrics import compute_hausdorff_distance

    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float("nan")
    p = torch.from_numpy(pred_mask[None, None].astype(np.uint8))
    g = torch.from_numpy(gt_mask[None, None].astype(np.uint8))
    try:
        val = compute_hausdorff_distance(p, g, include_background=True, percentile=95,
                                         spacing=list(spacing))
        return float(val.item())
    except Exception:
        return float("nan")


def evaluate_cases(
    networks: Sequence[torch.nn.Module],
    dataset: PreprocessedDataset,
    case_ids: Sequence[str],
    patch_size: Sequence[int],
    num_classes: int,
    regions: Dict[str, List[int]],
    class_names: Dict[int, str],
    device: torch.device,
    tile_step_size: float = 0.5,
    mirror_axes=(0, 1, 2),
    amp: bool = True,
    postprocess_kwargs: Dict = None,
    compute_hd: bool = True,
) -> pd.DataFrame:
    records = []
    for case_id in tqdm(case_ids, desc="cases"):
        data, seg, props = dataset.load_case(case_id)
        x = torch.from_numpy(np.asarray(data, dtype=np.float32))

        probs = None
        for net in networks:   # fold ensemble: average softmax, as nnU-Net does
            logits = predict_sliding_window(
                net, x, patch_size, num_classes, tile_step_size=tile_step_size,
                mirror_axes=mirror_axes, device=device, amp=amp,
            )
            p = torch.softmax(logits, 0)
            probs = p if probs is None else probs + p
        pred = (probs / len(networks)).argmax(0).cpu().numpy().astype(np.uint8)

        if postprocess_kwargs is not None:
            from src.evaluation.postprocessing import postprocess_prediction
            pred = postprocess_prediction(pred, **postprocess_kwargs)

        gt = np.asarray(seg[0])
        gt = np.where(gt < 0, 0, gt)   # -1 = outside the brain, scored as background

        rec = {"case_id": case_id}
        for name, labels in regions.items():
            pm, gm = np.isin(pred, labels), np.isin(gt, labels)
            denom = pm.sum() + gm.sum()
            rec[f"dice_{name}"] = float(2.0 * (pm & gm).sum() / denom) if denom > 0 else np.nan
            rec[f"vol_pred_{name}"] = int(pm.sum())
            rec[f"vol_true_{name}"] = int(gm.sum())
            if compute_hd:
                rec[f"hd95_{name}"] = hd95(pm, gm)
        for label, cname in class_names.items():
            pm, gm = pred == label, gt == label
            denom = pm.sum() + gm.sum()
            rec[f"dice_class_{cname}"] = float(2.0 * (pm & gm).sum() / denom) if denom > 0 else np.nan
        records.append(rec)
    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame, regions: Dict, title: str) -> Dict:
    table = Table(title=title, style="bold magenta")
    for col in ("Region", "Mean Dice", "Std", "Median", "Mean HD95"):
        table.add_column(col, justify="left" if col == "Region" else "right",
                         style="bold" if col == "Region" else None)
    summary = {}
    for name in regions:
        d = df[f"dice_{name}"]
        h = df.get(f"hd95_{name}")
        summary[f"dice_{name}"] = float(d.mean())
        summary[f"hd95_{name}"] = float(h.mean()) if h is not None else None
        table.add_row(name, f"{d.mean():.4f}", f"{d.std():.4f}", f"{d.median():.4f}",
                      f"{h.mean():.2f}" if h is not None else "-")
    summary["mean_region_dice"] = float(np.mean([summary[f"dice_{n}"] for n in regions]))
    table.add_row("[bold]Mean[/bold]", f"[bold]{summary['mean_region_dice']:.4f}[/bold]", "", "", "")
    console.print(table)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate nnU-Net replica runs")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="Fold run directories (runs/replica_*_fold0 ...)")
    ap.add_argument("--split", choices=["val", "test"], default="val",
                    help="val = per-fold out-of-fold CV; test = held-out set, folds ensembled")
    ap.add_argument("--checkpoint", choices=["best", "final", "latest"], default="best")
    ap.add_argument("--output_dir", default=None, help="Where to write CSV/JSON (default: first run dir)")
    ap.add_argument("--preprocessed_dir", help="Override replica.preprocessed_dir")
    ap.add_argument("--data_dir")
    ap.add_argument("--extra_data_dir", action="append", default=None)
    ap.add_argument("--postprocess", action="store_true", help="Force post-processing on")
    ap.add_argument("--no_tta", action="store_true", help="Disable mirroring TTA")
    ap.add_argument("--no_hd95", action="store_true", help="Skip HD95 (much faster)")
    ap.add_argument("--max_cases", type=int, default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["data"]["train_dir"] = args.data_dir
    if args.extra_data_dir:
        config["data"]["extra_train_dirs"] = args.extra_data_dir

    rcfg = ReplicaConfig.from_config(config)
    if args.preprocessed_dir:
        rcfg.preprocessed_dir = args.preprocessed_dir

    plan_cfg = Plans.load(rcfg.plans_json).get_configuration(rcfg.configuration)
    patch_size = list(rcfg.patch_size or plan_cfg.patch_size)
    num_classes = int(config["data"]["num_classes"])
    num_input_channels = len(config["data"]["modalities"])
    regions = config["evaluation"]["regions"]
    class_names = {int(k): v for k, v in config["data"]["class_names"].items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dirs = [Path(p) for p in args.run_dirs]
    missing = [p for p in run_dirs if not p.is_dir()]
    if missing:
        console.print(f"[red]run dir(s) not found: {missing}[/red]")
        sys.exit(1)

    postprocess_kwargs = None
    if args.postprocess or (config.get("inference", {}) or {}).get("postprocess", False):
        from src.evaluation.postprocessing import postprocess_kwargs_from_config
        postprocess_kwargs = postprocess_kwargs_from_config(config)
        console.print("[dim]post-processing: on[/dim]")

    mirror_axes = None if args.no_tta else (0, 1, 2)
    folds, test_ids, _ = build_replica_splits(config)
    dataset_folder = rcfg.resolved_preprocessed_dir()
    out_dir = Path(args.output_dir) if args.output_dir else run_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    def fold_of(run_dir: Path) -> int:
        ckpt = torch.load(str(run_dir / f"checkpoint_{args.checkpoint}.pth"),
                          map_location="cpu", weights_only=False)
        return int(ckpt.get("fold", 0))

    if args.split == "val":
        frames = []
        for run_dir in run_dirs:
            fold = fold_of(run_dir)
            case_ids = folds[fold][1]
            if args.max_cases:
                case_ids = case_ids[: args.max_cases]
            console.print(f"[bold]fold {fold}: {len(case_ids)} val cases "
                          f"({run_dir.name})[/bold]")
            net = load_fold_network(run_dir, args.checkpoint, plan_cfg,
                                    num_input_channels, num_classes, device)
            df = evaluate_cases(
                [net], PreprocessedDataset(dataset_folder, case_ids), case_ids, patch_size,
                num_classes, regions, class_names, device, rcfg.tile_step_size, mirror_axes,
                rcfg.amp, postprocess_kwargs, compute_hd=not args.no_hd95,
            )
            df.insert(0, "fold", fold)
            frames.append(df)
            del net
            torch.cuda.empty_cache()
        df_all = pd.concat(frames, ignore_index=True)
        title = f"Out-of-fold validation ({len(run_dirs)} fold(s), checkpoint_{args.checkpoint})"
    else:
        case_ids = test_ids[: args.max_cases] if args.max_cases else test_ids
        if not case_ids:
            console.print("[red]No held-out test cases — data.kfold_holdout_test is false[/red]")
            sys.exit(1)
        console.print(f"[bold]held-out test: {len(case_ids)} cases, "
                      f"{len(run_dirs)}-fold ensemble[/bold]")
        nets = [load_fold_network(rd, args.checkpoint, plan_cfg, num_input_channels,
                                  num_classes, device) for rd in run_dirs]
        df_all = evaluate_cases(
            nets, PreprocessedDataset(dataset_folder, case_ids), case_ids, patch_size,
            num_classes, regions, class_names, device, rcfg.tile_step_size, mirror_axes,
            rcfg.amp, postprocess_kwargs, compute_hd=not args.no_hd95,
        )
        title = f"Held-out test ({len(run_dirs)}-fold ensemble, checkpoint_{args.checkpoint})"

    summary = print_summary(df_all, regions, title)
    summary.update({
        "split": args.split, "checkpoint": args.checkpoint, "num_cases": len(df_all),
        "tta": mirror_axes is not None, "postprocess": postprocess_kwargs is not None,
        "run_dirs": [str(p) for p in run_dirs],
    })

    suffix = f"{args.split}_{args.checkpoint}" + ("_pp" if postprocess_kwargs else "")
    df_all.to_csv(out_dir / f"replica_metrics_{suffix}.csv", index=False)
    with open(out_dir / f"replica_summary_{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    console.print(f"[green]Wrote {out_dir}/replica_metrics_{suffix}.csv and "
                  f"replica_summary_{suffix}.json[/green]")


if __name__ == "__main__":
    main()
