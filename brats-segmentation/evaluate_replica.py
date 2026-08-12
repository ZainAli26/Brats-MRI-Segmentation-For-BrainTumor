#!/usr/bin/env python3
"""Evaluate replica runs: out-of-fold CV Dice/HD95, or a 5-fold ensemble on the test set.

    # per-fold validation (the CV metric) — pass every fold to pool them into one OOF number
    python evaluate_replica.py --config experiments/exp20_replica_resenc_m_11g_5fold.yaml \
        --run_dirs runs/replica_*_fold0 runs/replica_*_fold1 ... --split val

    # leak-free held-out test, 5-fold ensemble (softmax averaged across folds)
    python evaluate_replica.py --config <cfg> --run_dirs runs/replica_*_fold[0-4] --split test

    # nnU-Net post-processing: determine on OOF val, then apply the saved ops to test
    python evaluate_replica.py --config <cfg> --run_dirs runs/replica_*_fold[0-4] \
        --split val --determine_postprocessing            # writes postprocessing.json
    python evaluate_replica.py --config <cfg> --run_dirs runs/replica_*_fold[0-4] \
        --split test --postprocessing_json <out>/postprocessing.json

``--split val`` scores each fold's own validation cases with that fold's network, which is
what "out-of-fold" means and what the native `nnUNetv2_train` validation reports.
``--split test`` scores the seed-42 10 % held-out patients — never seen by any fold — with
all folds ensembled, so it is directly comparable to the native ensemble prediction.

Inference is the replica's own (Gaussian tiling + mirroring TTA over all 8 flips, logits
averaged). Two *different* post-processing families are available and must not be confused:

* ``--determine_postprocessing`` / ``--postprocessing_json`` — nnU-Net's own: greedily select
  keep-largest-component ops on the OOF predictions, apply the accepted ops verbatim to test.
  Determination only ever sees validation data, so there is no leakage. This is what
  ``nnUNetv2_determine_postprocessing`` does, and it is often a no-op.
* ``--postprocess`` (or ``inference.postprocess`` in the config) — the BraTS-competition
  heuristic with hand-picked thresholds (small-component cleanup, tiny-ET suppression, hole
  filling). nnU-Net does none of this. Useful as a comparison, not as "the nnU-Net number".
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


def score_case(
    pred: np.ndarray,
    gt: np.ndarray,
    case_id: str,
    regions: Dict[str, List[int]],
    class_names: Dict[int, str],
    compute_hd: bool = True,
) -> Dict:
    """Region + per-class Dice (and optional HD95) for one already-computed prediction."""
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
    return rec


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
    nnunet_ops: List[List[int]] = None,
    save_pred_dir: Path = None,
) -> pd.DataFrame:
    """Predict + score. Predictions are cached RAW (pre-post-processing) when asked.

    ``nnunet_ops`` (a determined connected-component op list) is applied before the
    optional heuristic ``postprocess_kwargs``, matching the order nnU-Net would use if
    both were in play. ``save_pred_dir`` stores raw argmax predictions so post-processing
    can be determined and re-scored later without re-running inference.
    """
    records = []
    if save_pred_dir is not None:
        save_pred_dir.mkdir(parents=True, exist_ok=True)

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

        if save_pred_dir is not None:
            np.save(save_pred_dir / f"{case_id}.npy", pred)

        if nnunet_ops:
            from src.evaluation.nnunet_postprocessing import apply_postprocessing
            pred = apply_postprocessing(pred, nnunet_ops)

        if postprocess_kwargs is not None:
            from src.evaluation.postprocessing import postprocess_prediction
            pred = postprocess_prediction(pred, **postprocess_kwargs)

        gt = np.asarray(seg[0])
        gt = np.where(gt < 0, 0, gt)   # -1 = outside the brain, scored as background

        records.append(score_case(pred, gt, case_id, regions, class_names, compute_hd))
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
    # nanmean, not mean: a region absent from every case in the split (RC is often absent)
    # scores NaN, and a plain mean would turn the whole headline number into NaN.
    per_region = [summary[f"dice_{n}"] for n in regions]
    scored = [v for v in per_region if not np.isnan(v)]
    summary["mean_region_dice"] = float(np.mean(scored)) if scored else float("nan")
    summary["regions_scored"] = [n for n in regions if not np.isnan(summary[f"dice_{n}"])]
    label = "[bold]Mean[/bold]"
    if len(scored) < len(per_region):
        label += f" [dim](over {len(scored)}/{len(per_region)} regions)[/dim]"
    table.add_row(label, f"[bold]{summary['mean_region_dice']:.4f}[/bold]", "", "", "")
    console.print(table)
    return summary


def determine_and_report(
    case_ids: Sequence[str],
    pred_cache: Path,
    dataset_folder: Path,
    out_dir: Path,
    num_classes: int,
    regions: Dict[str, List[int]],
    class_names: Dict[int, str],
    criterion: str,
    compute_hd: bool = True,
) -> None:
    """Run nnU-Net's post-processing determination on cached OOF predictions.

    Determination is a pure re-scoring loop over the cached predictions, so it costs no
    extra inference. The resulting ops go to ``postprocessing.json`` for later application
    to the held-out test set — determination itself only ever sees validation data.
    """
    from src.evaluation.nnunet_postprocessing import (
        apply_postprocessing, determine_postprocessing_streaming, num_determination_passes,
    )

    available = [c for c in case_ids if (pred_cache / f"{c}.npy").is_file()]
    if not available:
        console.print(f"[red]No cached predictions in {pred_cache}[/red]")
        return
    if len(available) < len(case_ids):
        console.print(f"[yellow]{len(case_ids) - len(available)} cached predictions missing; "
                      f"determining on {len(available)} cases[/yellow]")

    def load(case_id: str):
        pred = np.load(pred_cache / f"{case_id}.npy")
        seg = np.load(dataset_folder / f"{case_id}_seg.npy", mmap_mode="r")
        return pred, np.asarray(seg[0])

    foreground = list(range(1, num_classes))
    n_passes = num_determination_passes(foreground, regions, criterion)
    console.print(f"\n[bold]Determining nnU-Net post-processing on {len(available)} OOF cases "
                  f"(criterion={criterion}, {n_passes} scoring passes)[/bold]")
    bar = tqdm(total=len(available) * n_passes, desc="determine PP")
    ops, report = determine_postprocessing_streaming(
        available, load, foreground, regions, criterion=criterion,
        on_case=lambda: bar.update(1),
    )
    bar.close()

    table = Table(title="Post-processing determination (nnU-Net rules)", style="bold magenta")
    for col in ("Candidate", "Judged on", "Before", "After", "Region mean", "Decision"):
        table.add_column(col, justify="left" if col == "Candidate" else "right")
    for entry in report["log"]:
        table.add_row(
            entry["op"], entry["judged_on"],
            "-" if entry["judged_value_before"] is None else f"{entry['judged_value_before']:.4f}",
            "-" if entry["judged_value"] is None else f"{entry['judged_value']:.4f}",
            "-" if entry["region_mean"] is None else f"{entry['region_mean']:.4f}",
            "[green]ACCEPT[/green]" if entry["accepted"] else "[dim]reject[/dim]",
        )
    console.print(table)

    def fmt(v, plus=False):
        return "-" if v is None else (f"{v:+.4f}" if plus else f"{v:.4f}")

    console.print(f"selection metric ({criterion}): {fmt(report['baseline_dice'])} -> "
                  f"{fmt(report['final_dice'])}  |  BraTS region mean: "
                  f"{fmt(report['baseline_region_dice'])} -> {fmt(report['final_region_dice'])} "
                  f"({fmt(report['region_gain'], plus=True)})")

    pp_path = out_dir / "postprocessing.json"
    with open(pp_path, "w") as f:
        json.dump({"operations": ops, "regions": regions, "num_classes": num_classes,
                   "criterion": criterion, "report": report,
                   "n_val_cases": len(available)}, f, indent=2, default=float)
    console.print(f"[green]Wrote {pp_path}[/green]")

    if not ops:
        console.print("[dim]No candidate improved validation Dice — nnU-Net post-processing is "
                      "a no-op for this model. Report raw predictions.[/dim]")
        return

    # Re-score the OOF set with the accepted ops so the CV number is reported both ways.
    rows = []
    for case_id in tqdm(available, desc="rescoring with ops"):
        pred, gt = load(case_id)
        gt = np.where(gt < 0, 0, gt)
        rows.append(score_case(apply_postprocessing(pred, ops), gt, case_id,
                               regions, class_names, compute_hd))
    df_pp = pd.DataFrame(rows)
    summary_pp = print_summary(df_pp, regions, "Out-of-fold validation + nnU-Net post-processing")
    summary_pp["nnunet_postprocess_ops"] = ops
    df_pp.to_csv(out_dir / "replica_metrics_val_nnunet_pp.csv", index=False)
    with open(out_dir / "replica_summary_val_nnunet_pp.json", "w") as f:
        json.dump(summary_pp, f, indent=2, default=float)
    console.print(f"[green]Wrote {out_dir}/replica_metrics_val_nnunet_pp.csv[/green]")
    console.print(f"[bold]Apply to test:[/bold] --split test --postprocessing_json {pp_path}")


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
    ap.add_argument("--postprocess", action="store_true",
                    help="Apply the BraTS heuristic cleanup (fixed thresholds: small-component "
                         "removal, tiny-ET suppression, hole filling). NOT what nnU-Net does — "
                         "see --determine_postprocessing for that.")
    ap.add_argument("--determine_postprocessing", action="store_true",
                    help="nnU-Net's data-driven post-processing: greedily select "
                         "keep-largest-component ops on the OOF predictions and write "
                         "postprocessing.json. Requires --split val (determining on test leaks).")
    ap.add_argument("--postprocessing_json",
                    help="Apply previously determined nnU-Net ops (use on --split test)")
    ap.add_argument("--pp_criterion", choices=["labels", "regions"], default="labels",
                    help="Metric driving op selection: 'labels' = nnU-Net-faithful, "
                         "'regions' = the BraTS ET/TC/WT/RC score that is actually reported")
    ap.add_argument("--pred_cache_dir", default=None,
                    help="Where raw predictions are cached (default: <output_dir>/predictions_<split>)")
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

    inf_cfg = config.get("inference", {}) or {}

    postprocess_kwargs = None
    if args.postprocess or inf_cfg.get("postprocess", False):
        from src.evaluation.postprocessing import postprocess_kwargs_from_config
        postprocess_kwargs = postprocess_kwargs_from_config(config)
        console.print("[dim]heuristic BraTS post-processing: on (not nnU-Net's)[/dim]")

    nnunet_ops = None
    if args.postprocessing_json:
        with open(args.postprocessing_json) as f:
            nnunet_ops = json.load(f).get("operations", [])
        console.print(f"[cyan]nnU-Net post-processing ops: {nnunet_ops or 'none (determined as no-op)'}[/cyan]")
    if args.determine_postprocessing and args.split != "val":
        console.print("[red]--determine_postprocessing requires --split val: determining on the "
                      "test set leaks it. Determine on val, then pass the JSON to --split test.[/red]")
        sys.exit(1)
    if args.determine_postprocessing and args.postprocessing_json:
        console.print("[red]--determine_postprocessing and --postprocessing_json are mutually "
                      "exclusive: determination would start from the raw predictions while the "
                      "reported metrics already had the supplied ops applied.[/red]")
        sys.exit(1)
    if args.determine_postprocessing and postprocess_kwargs is not None:
        console.print("[yellow]Determining nnU-Net ops on top of heuristic post-processing — "
                      "the ops will encode whatever the heuristic left behind.[/yellow]")

    # TTA: nnU-Net mirrors over all three spatial axes (2^3 = 8 forward passes per tile).
    # Honour the config's inference.tta so a run configured without TTA is not silently given it.
    tta_enabled = bool(inf_cfg.get("tta", True)) and not args.no_tta
    mirror_axes = (0, 1, 2) if tta_enabled else None
    console.print(f"[dim]mirroring TTA: {'on (8 passes/tile)' if tta_enabled else 'off'}[/dim]")

    folds, test_ids, _ = build_replica_splits(config)
    dataset_folder = rcfg.resolved_preprocessed_dir()
    out_dir = Path(args.output_dir) if args.output_dir else run_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_cache = (Path(args.pred_cache_dir) if args.pred_cache_dir
                  else out_dir / f"predictions_{args.split}")

    def fold_of(run_dir: Path) -> int:
        ckpt = torch.load(str(run_dir / f"checkpoint_{args.checkpoint}.pth"),
                          map_location="cpu", weights_only=False)
        return int(ckpt.get("fold", 0))

    # Cache raw predictions whenever we may need to re-score them without re-inference.
    save_pred_dir = pred_cache if args.determine_postprocessing else None
    scored_ids: List[str] = []

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
                nnunet_ops=nnunet_ops, save_pred_dir=save_pred_dir,
            )
            df.insert(0, "fold", fold)
            frames.append(df)
            scored_ids.extend(case_ids)
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
            nnunet_ops=nnunet_ops, save_pred_dir=save_pred_dir,
        )
        scored_ids = list(case_ids)
        title = f"Held-out test ({len(run_dirs)}-fold ensemble, checkpoint_{args.checkpoint})"

    summary = print_summary(df_all, regions, title)
    summary.update({
        "split": args.split, "checkpoint": args.checkpoint, "num_cases": len(df_all),
        "tta": mirror_axes is not None,
        "heuristic_postprocess": postprocess_kwargs is not None,
        "nnunet_postprocess_ops": nnunet_ops,
        "run_dirs": [str(p) for p in run_dirs],
    })

    suffix = f"{args.split}_{args.checkpoint}"
    if postprocess_kwargs:
        suffix += "_pp"
    if nnunet_ops:
        suffix += "_nnpp"
    df_all.to_csv(out_dir / f"replica_metrics_{suffix}.csv", index=False)
    with open(out_dir / f"replica_summary_{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    console.print(f"[green]Wrote {out_dir}/replica_metrics_{suffix}.csv and "
                  f"replica_summary_{suffix}.json[/green]")

    if args.determine_postprocessing:
        determine_and_report(
            scored_ids, pred_cache, Path(dataset_folder), out_dir, num_classes, regions,
            class_names, args.pp_criterion, compute_hd=not args.no_hd95,
        )


if __name__ == "__main__":
    main()
