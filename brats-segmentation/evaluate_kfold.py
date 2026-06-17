#!/usr/bin/env python3
"""BraTS K-Fold Ensemble Evaluation with Post-Processing.

Evaluates exp18-style k-fold models.  Supports:
  - Single fold evaluation
  - Softmax ensemble across all available folds
  - Test-Time Augmentation (flip ensemble)
  - Post-processing comparison (with / without)

Usage:
    # Evaluate fold0 checkpoint (no ensemble)
    python evaluate_kfold.py \\
        --fold_dirs runs/nnunet_v2_20260507_021354_fold0 \\
        --config experiments/exp18_nnunet_v2_residual_5fold.yaml

    # Ensemble all 5 folds (pass all fold dirs)
    python evaluate_kfold.py \\
        --fold_dirs runs/fold0 runs/fold1 runs/fold2 runs/fold3 runs/fold4 \\
        --config experiments/exp18_nnunet_v2_residual_5fold.yaml

    # Enable TTA (flip ensemble, ~8× slower)
    python evaluate_kfold.py --fold_dirs runs/fold0 --config ... --tta

    # Evaluate on val instead of the full dataset
    python evaluate_kfold.py --fold_dirs runs/fold0 --config ... --split val_fold0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.data.dataset import build_file_list
from src.data.preprocessing import get_val_transforms
from src.data.splits import create_kfold_splits
from src.evaluation.postprocessing import postprocess_prediction
from src.models.factory import create_model
from src.utils import inference_wrapper
from src.utils.experiment import load_config

console = Console()


# ── inference helpers ─────────────────────────────────────────────────────────

def _sliding_window_softmax(model, images, spatial_size, sw_batch, sw_overlap, use_amp, device):
    """Run sliding-window inference and return raw softmax logits (B, C, H, W, D)."""
    with torch.no_grad():
        with autocast(enabled=use_amp):
            logits = sliding_window_inference(
                images, spatial_size, sw_batch,
                inference_wrapper(model), overlap=sw_overlap,
            )
    return torch.softmax(logits, dim=1)


def _tta_softmax(model, images, spatial_size, sw_batch, sw_overlap, use_amp, device):
    """Average softmax over 8 flip augmentations (all axis combinations)."""
    dims_list = [
        [],
        [2], [3], [4],
        [2, 3], [2, 4], [3, 4],
        [2, 3, 4],
    ]
    acc = None
    for dims in dims_list:
        aug = torch.flip(images, dims) if dims else images
        prob = _sliding_window_softmax(model, aug, spatial_size, sw_batch, sw_overlap, use_amp, device)
        if dims:
            prob = torch.flip(prob, dims)
        acc = prob if acc is None else acc + prob
    return acc / len(dims_list)


# ── region dice / hd95 ───────────────────────────────────────────────────────

def _region_metrics(pred_oh, lab_oh, regions, device):
    """Compute per-region Dice (and HD95 where possible) from one-hot tensors."""
    result = {}
    pred_argmax = pred_oh.argmax(dim=0)
    lab_argmax = lab_oh.argmax(dim=0)

    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

    for region_name, label_indices in regions.items():
        pred_r = torch.zeros_like(pred_argmax, dtype=torch.bool)
        lab_r = torch.zeros_like(lab_argmax, dtype=torch.bool)
        for idx in label_indices:
            pred_r |= (pred_argmax == idx)
            lab_r |= (lab_argmax == idx)

        intersection = (pred_r & lab_r).sum().float()
        union = pred_r.sum().float() + lab_r.sum().float()
        dice = (2.0 * intersection / (union + 1e-7)).item()
        result[f"dice_{region_name}"] = dice

        # HD95 — skip if one mask is empty (undefined)
        if pred_r.sum() > 0 and lab_r.sum() > 0:
            # Shape: (1, 1, H, W, D)
            p = pred_r.unsqueeze(0).unsqueeze(0).float().to(device)
            l = lab_r.unsqueeze(0).unsqueeze(0).float().to(device)
            hd_metric.reset()
            try:
                hd_metric(y_pred=p, y=l)
                hd = hd_metric.aggregate().item()
            except Exception:
                hd = float("nan")
        else:
            hd = float("nan")
        result[f"hd95_{region_name}"] = hd

        result[f"vol_pred_{region_name}"] = int(pred_r.sum().item())
        result[f"vol_true_{region_name}"] = int(lab_r.sum().item())

    return result


# ── core evaluation loop ──────────────────────────────────────────────────────

def evaluate(
    models,
    dataloader,
    config: dict,
    device: torch.device,
    use_tta: bool = False,
    use_postproc: bool = False,
    postproc_kwargs: dict = None,
) -> pd.DataFrame:
    """Run evaluation over dataloader, optionally ensembling multiple models.

    Args:
        models: list of torch.nn.Module (already on device, eval mode).
        use_tta: whether to apply flip TTA per model.
        use_postproc: whether to apply CCA + ET suppression + hole filling.
        postproc_kwargs: forwarded to postprocess_prediction().
    """
    if postproc_kwargs is None:
        postproc_kwargs = {}

    num_classes = config["data"]["num_classes"]
    spatial_size = config["preprocessing"]["spatial_size"]
    sw_batch = config["training"]["sw_batch_size"]
    sw_overlap = config["training"]["sw_overlap"]
    regions = config["evaluation"]["regions"]
    use_amp = config["training"]["amp"] and device.type == "cuda"

    post_pred_discrete = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    dice_metric = DiceMetric(include_background=False, reduction="none")
    records = []

    for batch_data in tqdm(dataloader, desc="Evaluating"):
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        case_id = batch_data.get("case_id", ["unknown"])[0]

        # ── ensemble softmax across all fold models ──
        prob_acc = None
        for m in models:
            fn = _tta_softmax if use_tta else _sliding_window_softmax
            prob = fn(m, images, spatial_size, sw_batch, sw_overlap, use_amp, device)
            prob_acc = prob if prob_acc is None else prob_acc + prob
        ensemble_prob = prob_acc / len(models)  # (B, C, H, W, D)

        # ── optional post-processing ──
        if use_postproc:
            # Convert to argmax numpy, post-process, convert back to one-hot
            argmax_np = ensemble_prob[0].argmax(dim=0).cpu().numpy().astype(np.int32)
            pp_np = postprocess_prediction(argmax_np, **postproc_kwargs)
            pp_t = torch.from_numpy(pp_np).long().to(device)
            # Re-build one-hot from post-processed argmax
            pred_oh = torch.zeros(num_classes, *pp_t.shape, device=device)
            pred_oh.scatter_(0, pp_t.unsqueeze(0), 1)
        else:
            outputs_list = decollate_batch(ensemble_prob)
            pred_oh = post_pred_discrete(outputs_list[0])

        labels_list = decollate_batch(labels)
        lab_oh = post_label(labels_list[0])

        # Per-class Dice
        dice_metric.reset()
        dice_metric(y_pred=[pred_oh], y=[lab_oh])
        class_dice = dice_metric.aggregate().cpu().numpy().flatten()

        record = {"case_id": case_id}
        for i, cname in enumerate(["NCR", "ED", "ET"]):
            record[f"dice_{cname}"] = float(class_dice[i]) if i < len(class_dice) else np.nan

        # Per-region Dice + HD95
        record.update(_region_metrics(pred_oh, lab_oh, regions, device))
        records.append(record)

    return pd.DataFrame(records)


# ── summary printing ──────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, label: str, config: dict):
    regions = list(config["evaluation"]["regions"].keys())
    class_names = ["NCR", "ED", "ET"]

    table = Table(title=f"[bold]{label}[/bold]", style="cyan")
    table.add_column("Region/Class", style="bold")
    table.add_column("Mean Dice", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Mean HD95", justify="right")

    for rname in regions:
        col = f"dice_{rname}"
        hd_col = f"hd95_{rname}"
        if col in df:
            hd = df[hd_col].mean() if hd_col in df else float("nan")
            table.add_row(
                f"[magenta]{rname}[/magenta]",
                f"{df[col].mean():.4f}", f"{df[col].std():.4f}",
                f"{df[col].median():.4f}", f"{hd:.2f}",
            )

    mean_region = np.mean([df[f"dice_{r}"].mean() for r in regions if f"dice_{r}" in df])
    table.add_row("", "", "", "", "")
    table.add_row("[bold green]Mean Region[/bold green]", f"[bold green]{mean_region:.4f}[/bold green]", "", "", "")

    for cname in class_names:
        col = f"dice_{cname}"
        if col in df:
            table.add_row(f"  {cname}", f"{df[col].mean():.4f}", f"{df[col].std():.4f}", "", "")

    console.print(table)
    return mean_region


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="K-Fold ensemble evaluation for BraTS")
    parser.add_argument("--fold_dirs", nargs="+", required=True,
                        help="One or more fold run directories containing best_model.pth")
    parser.add_argument("--config", required=True,
                        help="Experiment config YAML (e.g. experiments/exp18_...yaml)")
    parser.add_argument("--split", choices=["all", "val_fold0", "val_fold1", "val_fold2",
                                             "val_fold3", "val_fold4"],
                        default="all",
                        help="Which cases to evaluate on. 'all' uses the full training dataset.")
    parser.add_argument("--tta", action="store_true",
                        help="Enable flip Test-Time Augmentation (8 flips, ~8× slower)")
    parser.add_argument("--no_postproc", action="store_true",
                        help="Skip post-processing (run raw model output only)")
    parser.add_argument("--et_min_voxels", type=int, default=250,
                        help="ET suppression threshold in voxels (default: 250)")
    parser.add_argument("--min_component_size", type=int, default=50,
                        help="CCA min component size in voxels (default: 50)")
    parser.add_argument("--no_fill_holes", action="store_true",
                        help="Disable morphological hole filling")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save CSV results (default: first fold_dir/eval_kfold)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data.train_dir from config (useful inside Docker)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["data"]["train_dir"] = args.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_dirs = [Path(d).expanduser() for d in args.fold_dirs]
    output_dir = Path(args.output_dir) if args.output_dir else fold_dirs[0] / "eval_kfold"
    output_dir.mkdir(parents=True, exist_ok=True)

    postproc_kwargs = dict(
        et_min_voxels=args.et_min_voxels,
        min_component_size=args.min_component_size,
        fill_holes=not args.no_fill_holes,
    )

    n_folds = config["data"].get("n_folds", 5)
    console.print(Panel.fit(
        f"[bold cyan]K-Fold Ensemble Evaluation — Exp 18[/bold cyan]\n"
        f"[dim]Folds loaded: {len(fold_dirs)} / {n_folds} | "
        f"TTA: {args.tta} | Post-proc: {not args.no_postproc} | "
        f"Device: {device}[/dim]",
        border_style="bright_blue",
    ))

    # ── resolve which cases to evaluate ──────────────────────────────────────
    data_dir = Path(config["data"]["train_dir"]).expanduser()
    if not data_dir.exists():
        console.print(f"[red]Data dir not found: {data_dir}[/red]")
        sys.exit(1)

    folds = create_kfold_splits(str(data_dir), n_folds=n_folds, seed=config["data"]["split_seed"])

    if args.split == "all":
        # Every case in the dataset
        all_cases = sorted(set(c for tc, vc in folds for c in (tc + vc)))
        eval_cases = sorted(all_cases, key=lambda p: p.name)
        console.print(f"[dim]Evaluating on all {len(eval_cases)} cases[/dim]")
    else:
        fold_idx = int(args.split.split("fold")[-1])
        _, eval_cases = folds[fold_idx]
        console.print(f"[dim]Evaluating on fold {fold_idx} val set: {len(eval_cases)} cases[/dim]")

    # ── dataloader ────────────────────────────────────────────────────────────
    modalities = config["data"]["modalities"]
    label_map = {int(k): int(v) for k, v in config["data"]["label_map"].items()}
    spatial_size = config["preprocessing"]["spatial_size"]

    val_transform = get_val_transforms(spatial_size, modalities, label_map)
    file_list = build_file_list(eval_cases, modalities, include_label=True)
    eval_ds = Dataset(file_list, transform=val_transform)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    console.print(f"[dim]Cases in DataLoader: {len(eval_ds)}[/dim]\n")

    # ── load fold models ──────────────────────────────────────────────────────
    models = []
    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / "best_model.pth"
        if not ckpt_path.exists():
            console.print(f"[yellow]No checkpoint in {fold_dir}, skipping[/yellow]")
            continue
        m = create_model(config)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device).eval()
        epoch = ckpt.get("epoch", "?")
        val_dice = ckpt.get("val_dice", float("nan"))
        console.print(f"  [green]Loaded {fold_dir.name}[/green] — epoch {epoch}, val dice {val_dice:.4f}")
        models.append(m)

    if not models:
        console.print("[red bold]No valid checkpoints found. Exiting.[/red bold]")
        sys.exit(1)

    console.print(f"\n[bold]Ensembling {len(models)} model(s)[/bold]\n")

    # ── run WITHOUT post-processing ───────────────────────────────────────────
    console.print("[bold]Pass 1 — Raw (no post-processing)[/bold]")
    df_raw = evaluate(models, eval_loader, config, device,
                      use_tta=args.tta, use_postproc=False)
    mean_raw = print_summary(df_raw, "Raw Ensemble", config)

    # ── run WITH post-processing ──────────────────────────────────────────────
    results = {"raw": df_raw}
    mean_pp = None

    if not args.no_postproc:
        console.print(f"\n[bold]Pass 2 — With post-processing "
                      f"(ET<{args.et_min_voxels}vx→NCR, CCA≥{args.min_component_size}vx, "
                      f"hole_fill={not args.no_fill_holes})[/bold]")
        df_pp = evaluate(models, eval_loader, config, device,
                         use_tta=args.tta, use_postproc=True,
                         postproc_kwargs=postproc_kwargs)
        mean_pp = print_summary(df_pp, "Post-Processed", config)
        results["postprocessed"] = df_pp

        # Delta summary
        regions = list(config["evaluation"]["regions"].keys())
        delta_table = Table(title="Post-Processing Delta", style="bold yellow")
        delta_table.add_column("Region", style="bold")
        delta_table.add_column("Raw", justify="right")
        delta_table.add_column("Post-Proc", justify="right")
        delta_table.add_column("Δ", justify="right")
        for r in regions:
            col = f"dice_{r}"
            if col in df_raw and col in df_pp:
                raw_val = df_raw[col].mean()
                pp_val = df_pp[col].mean()
                delta = pp_val - raw_val
                color = "green" if delta >= 0 else "red"
                delta_table.add_row(
                    r, f"{raw_val:.4f}", f"{pp_val:.4f}",
                    f"[{color}]{delta:+.4f}[/{color}]",
                )
        console.print(delta_table)

    # ── save results ──────────────────────────────────────────────────────────
    for tag, df in results.items():
        csv_path = output_dir / f"metrics_{tag}.csv"
        df.to_csv(csv_path, index=False)
        console.print(f"[dim]Saved {csv_path}[/dim]")

    # JSON summary
    regions = list(config["evaluation"]["regions"].keys())
    summary = {
        "n_folds_ensembled": len(models),
        "fold_dirs": [str(d) for d in fold_dirs],
        "split": args.split,
        "tta": args.tta,
        "postproc_kwargs": postproc_kwargs,
        "raw": {
            "mean_region_dice": float(mean_raw),
            **{r: float(df_raw[f"dice_{r}"].mean()) for r in regions if f"dice_{r}" in df_raw},
        },
    }
    if mean_pp is not None:
        summary["postprocessed"] = {
            "mean_region_dice": float(mean_pp),
            **{r: float(df_pp[f"dice_{r}"].mean()) for r in regions if f"dice_{r}" in df_pp},
        }
    json_path = output_dir / "eval_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[bold green]Done. Results saved to {output_dir}[/bold green]")
    console.print(f"  Raw mean region Dice   : {mean_raw:.4f}")
    if mean_pp is not None:
        delta = mean_pp - mean_raw
        console.print(f"  Post-proc mean region  : {mean_pp:.4f}  (Δ {delta:+.4f})")


if __name__ == "__main__":
    main()
