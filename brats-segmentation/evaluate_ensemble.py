#!/usr/bin/env python3
"""Ensemble *different models* — the cross-experiment analogue of ``nnUNetv2_ensemble``.

``evaluate_replica.py`` ensembles the 5 folds of ONE experiment. This script ensembles
across experiments: ResEnc-M (exp20) + 5-channel subtraction (exp21) + SegResNet-DS (exp25)
+ … each contributing all of its folds. nnU-Net ensembles by averaging the per-configuration
softmax probabilities and argmaxing once at the end, and so does this.

    # held-out test set, three models x 5 folds each, equal weights
    python evaluate_ensemble.py --split test \
        --member experiments/exp20_replica_resenc_m_11g_5fold.yaml runs/exp20_fold[0-4] \
        --member experiments/exp21_replica_5ch_subtraction_5fold.yaml runs/exp21_fold[0-4] \
        --member experiments/exp25_replica_segresnet_ds_5fold.yaml runs/exp25_fold[0-4]

    # out-of-fold CV of the same ensemble (each case scored by the fold that held it out)
    python evaluate_ensemble.py --split val --member ... --member ... \
        --determine_postprocessing

    # then apply the determined post-processing to test
    python evaluate_ensemble.py --split test --member ... --member ... \
        --postprocessing_json <out>/postprocessing.json

Members may differ in architecture, plan, patch size and **input channels** — a 4-channel
model and a 5-channel subtraction model ensemble fine. Two invariants are enforced rather
than assumed:

* **Identical patient split.** Every member must produce the same per-fold validation case
  lists and the same held-out test ids, or the ensemble would be scoring one member on data
  another trained on. Differing *train* sets are fine and expected (exp27 adds synthetic
  cases to train only), so only the val/test sides are compared.
* **Identical label scheme.** Same ``num_classes``, class names and scoring regions.

Averaging happens in the **original uncropped volume**, not in preprocessed space. Members
with different input channels get different nonzero-crop bounding boxes, so their cropped
predictions have different shapes and are not voxel-aligned; un-cropping first is what makes
them comparable. Outside a member's crop the probability mass is put on background, which is
what the crop asserted about that region anyway.

``--split val`` is the honest CV number: for each case only the fold that held that case out
contributes, per member. ``--split test`` uses every fold of every member.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from evaluate_replica import (
    determine_and_report, load_fold_network, print_summary, score_case,
)
from src.nnunet_replica.config import ReplicaConfig
from src.nnunet_replica.dataloading import PreprocessedDataset
from src.nnunet_replica.inference import predict_sliding_window
from src.nnunet_replica.plans import Plans
from src.nnunet_replica.splits import build_replica_splits
from src.utils.experiment import load_config

console = Console()


# ── members ───────────────────────────────────────────────────────────────────

@dataclass
class Member:
    """One model in the ensemble: a config plus the fold run dirs trained from it."""
    name: str
    config: Dict
    rcfg: ReplicaConfig
    plan_cfg: object
    patch_size: List[int]
    num_input_channels: int
    dataset_folder: Path
    run_dirs: List[Path]
    folds: List[Tuple[List[str], List[str]]]
    test_ids: List[str]
    weight: float = 1.0
    # Fold indices this member supplies, read from the checkpoints at describe time so the
    # consistency checks and the member table work before any weights are loaded.
    ckpt_folds: List[int] = field(default_factory=list)
    # (fold index, network) pairs, held on CPU and moved to the GPU per case.
    nets: List[Tuple[int, torch.nn.Module]] = field(default_factory=list)
    ds: Optional[PreprocessedDataset] = None

    @property
    def fold_ids(self) -> List[int]:
        return list(self.ckpt_folds)


def fold_of_checkpoint(run_dir: Path, checkpoint: str) -> int:
    ckpt = torch.load(str(run_dir / f"checkpoint_{checkpoint}.pth"),
                      map_location="cpu", weights_only=False)
    return int(ckpt.get("fold", 0))


def describe_member(
    config_path: str,
    run_dirs: Sequence[str],
    checkpoint: str,
    weight: float,
    preprocessed_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Member:
    """Build a Member from its config and splits, *without* loading any weights.

    Kept separate from ``attach_networks`` so the cross-member consistency checks run before
    minutes of checkpoint loading — and so a mismatched label scheme is reported as such
    rather than as a state-dict shape error.
    """
    config = load_config(config_path)
    if data_dir:
        config["data"]["train_dir"] = data_dir
    rcfg = ReplicaConfig.from_config(config)
    if preprocessed_dir:
        rcfg.preprocessed_dir = preprocessed_dir

    plan_cfg = Plans.load(rcfg.plans_json).get_configuration(rcfg.configuration)
    folds, test_ids, _ = build_replica_splits(config)

    dirs = [Path(p) for p in run_dirs]
    name = Path(config_path).stem
    missing = [p for p in dirs if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"run dir(s) not found for {name}: {missing}")
    no_ckpt = [p for p in dirs if not (p / f"checkpoint_{checkpoint}.pth").is_file()]
    if no_ckpt:
        raise FileNotFoundError(
            f"{name}: no checkpoint_{checkpoint}.pth in {[str(p) for p in no_ckpt]}")

    member = Member(
        name=name, config=config, rcfg=rcfg, plan_cfg=plan_cfg,
        patch_size=list(rcfg.patch_size or plan_cfg.patch_size),
        num_input_channels=len(config["data"]["modalities"]),
        dataset_folder=rcfg.resolved_preprocessed_dir(), run_dirs=dirs,
        folds=folds, test_ids=test_ids, weight=weight,
    )
    member.ckpt_folds = [fold_of_checkpoint(d, checkpoint) for d in dirs]
    dupes = sorted({f for f in member.ckpt_folds if member.ckpt_folds.count(f) > 1})
    if dupes:
        raise ValueError(
            f"{name}: fold(s) {dupes} supplied more than once. Duplicated folds would be "
            f"double-weighted in the ensemble and, on --split val, would score a case with "
            f"a network that trained on it."
        )
    return member


def attach_networks(member: Member, checkpoint: str, num_classes: int) -> None:
    """Load the member's fold checkpoints onto the CPU."""
    for run_dir in member.run_dirs:
        fold = fold_of_checkpoint(run_dir, checkpoint)
        try:
            net = load_fold_network(run_dir, checkpoint, member.plan_cfg,
                                    member.num_input_channels, num_classes,
                                    torch.device("cpu"))
        except RuntimeError as exc:
            raise ValueError(
                f"{member.name}: {run_dir.name}'s checkpoint does not fit the network its "
                f"config describes ({member.num_input_channels} input channels, "
                f"{num_classes} classes, plan {member.rcfg.plans_json}). Check that the "
                f"config paired with these run dirs is the one they were trained from.\n{exc}"
            ) from exc
        member.nets.append((fold, net))


def verify_members(members: Sequence[Member]) -> Tuple[int, Dict, Dict]:
    """Enforce identical split and label scheme; return (num_classes, regions, class_names).

    The val/test partition must match exactly across members — otherwise a member could be
    predicting a case it trained on and the ensemble score would be meaningless. Train sets
    are deliberately *not* compared: exp27 adds synthetic cases to train only.
    """
    ref = members[0]
    ref_classes = int(ref.config["data"]["num_classes"])
    ref_regions = ref.config["evaluation"]["regions"]
    ref_names = {int(k): v for k, v in ref.config["data"]["class_names"].items()}
    ref_val = [sorted(va) for _, va in ref.folds]
    ref_test = sorted(ref.test_ids)

    for m in members[1:]:
        if int(m.config["data"]["num_classes"]) != ref_classes:
            raise ValueError(f"{m.name} has num_classes={m.config['data']['num_classes']}, "
                             f"{ref.name} has {ref_classes}. Cannot ensemble different label "
                             f"schemes — the softmax channels do not mean the same thing.")
        if m.config["evaluation"]["regions"] != ref_regions:
            raise ValueError(f"{m.name} scores different regions than {ref.name}: "
                             f"{m.config['evaluation']['regions']} vs {ref_regions}")
        names = {int(k): v for k, v in m.config["data"]["class_names"].items()}
        if names != ref_names:
            raise ValueError(f"{m.name} has different class names than {ref.name}: "
                             f"{names} vs {ref_names}")
        if len(m.folds) != len(ref.folds):
            raise ValueError(f"{m.name} has {len(m.folds)} folds, {ref.name} has {len(ref.folds)}")
        for i, (_, va) in enumerate(m.folds):
            if sorted(va) != ref_val[i]:
                extra = sorted(set(va) - set(ref_val[i]))[:3]
                missing = sorted(set(ref_val[i]) - set(va))[:3]
                raise ValueError(
                    f"{m.name} fold {i} validation set differs from {ref.name}'s "
                    f"({len(va)} vs {len(ref_val[i])} cases; e.g. only in {m.name}: {extra}, "
                    f"only in {ref.name}: {missing}). The ensemble would score a case with a "
                    f"network that trained on it. Align data.split_seed / n_folds / "
                    f"train_dir / extra_train_dirs across members."
                )
        if sorted(m.test_ids) != ref_test:
            raise ValueError(f"{m.name} has a different held-out test set than {ref.name} "
                             f"({len(m.test_ids)} vs {len(ref_test)} cases)")
    return ref_classes, ref_regions, ref_names


def print_members(members: Sequence[Member], split: str) -> None:
    table = Table(title=f"Ensemble members ({split})", style="bold cyan")
    for col in ("Member", "Folds", "In ch", "Patch", "Arch", "Weight", "Preprocessed cache"):
        table.add_column(col, justify="left" if col in ("Member", "Preprocessed cache") else "right")
    total_w = sum(m.weight for m in members)
    for m in members:
        arch = m.rcfg.architecture_override or m.plan_cfg.architecture.short_class_name
        table.add_row(m.name, ",".join(str(f) for f in sorted(m.fold_ids)),
                      str(m.num_input_channels), "x".join(str(p) for p in m.patch_size),
                      arch, f"{m.weight / total_w:.3f}", str(m.dataset_folder))
    console.print(table)


# ── geometry ──────────────────────────────────────────────────────────────────

def _crop_slicer(props: Dict) -> Tuple[slice, ...]:
    return tuple(slice(lo, hi) for lo, hi in props["bbox_used_for_cropping"])


def uncrop_probabilities(probs: np.ndarray, props: Dict) -> np.ndarray:
    """Place cropped (C, x, y, z) probabilities into the original uncropped volume.

    Outside the crop the mass goes entirely on background (channel 0). The crop bbox is the
    nonzero-intensity bounding box, so that region held no image signal and no label; giving
    it a flat zero vector instead would let whichever member cropped more loosely decide the
    argmax there on its own.
    """
    full = np.zeros((probs.shape[0], *props["shape_before_cropping"]), dtype=np.float32)
    full[0] = 1.0
    full[(slice(None), *_crop_slicer(props))] = probs
    return full


def uncrop_segmentation(seg_cropped: np.ndarray, props: Dict) -> np.ndarray:
    """Ground truth back in the original volume; -1 (outside brain) and outside-crop -> 0."""
    full = np.zeros(props["shape_before_cropping"], dtype=np.uint8)
    seg = np.asarray(seg_cropped)
    full[_crop_slicer(props)] = np.where(seg < 0, 0, seg).astype(np.uint8)
    return full


def check_geometry(members: Sequence[Member], case_id: str) -> Dict:
    """Return the reference properties for a case, asserting members agree on raw shape."""
    ref_props = None
    ref_member = None
    for m in members:
        props = m.ds.load_properties(case_id)
        if ref_props is None:
            ref_props, ref_member = props, m
        elif list(props["shape_before_cropping"]) != list(ref_props["shape_before_cropping"]):
            raise ValueError(
                f"{case_id}: {m.name} preprocessed a raw volume of shape "
                f"{list(props['shape_before_cropping'])} but {ref_member.name} saw "
                f"{list(ref_props['shape_before_cropping'])}. The members were built from "
                f"different source data for this case, so their predictions cannot be aligned."
            )
    return ref_props


# ── ensembling ────────────────────────────────────────────────────────────────

def ensemble_case(
    case_id: str,
    members: Sequence[Member],
    nets_for: Dict[str, List[Tuple[int, torch.nn.Module]]],
    num_classes: int,
    device: torch.device,
    mirror_axes,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted-average the members' softmax in original-volume space; return (pred, gt).

    Within a member its folds are averaged equally, then members are combined by weight, so
    a member's influence does not depend on how many folds it contributed. All folds of one
    member share a preprocessed cache and therefore a crop bbox, so their probabilities are
    summed in cheap cropped space and un-cropped once per member rather than once per fold.
    """
    ref_props = check_geometry(members, case_id)
    prob_sum = np.zeros((num_classes, *ref_props["shape_before_cropping"]), dtype=np.float32)
    weight_sum = 0.0
    gt = None

    for m in members:
        nets = nets_for[m.name]
        if not nets:
            continue
        data, seg, props = m.ds.load_case(case_id)
        if gt is None:
            gt = uncrop_segmentation(seg[0], props)
        x = torch.from_numpy(np.asarray(data, dtype=np.float32))
        if x.shape[0] != m.num_input_channels:
            raise ValueError(
                f"{case_id}: {m.name} expects {m.num_input_channels} input channels but its "
                f"cache at {m.dataset_folder} has {x.shape[0]}. The cache was built for a "
                f"different modality list — re-run preprocess_replica.py for this config."
            )

        cropped = None
        for _, net in nets:
            # Weights live on the CPU between networks: 5 folds x 3 members of a 100M-param
            # net will not co-reside on an 8 GB card. The transfer costs milliseconds against
            # seconds of 8-flip sliding-window inference.
            net.to(device)
            try:
                logits = predict_sliding_window(
                    net, x, m.patch_size, num_classes, tile_step_size=m.rcfg.tile_step_size,
                    mirror_axes=mirror_axes, device=device, amp=m.rcfg.amp,
                )
                probs = torch.softmax(logits, 0).cpu().numpy()
                del logits
            finally:
                net.to("cpu")
            cropped = probs if cropped is None else cropped + probs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        cropped /= len(nets)
        full = uncrop_probabilities(cropped, props)
        full *= m.weight                 # scale and accumulate in place: no 178 MB temporary
        prob_sum += full
        weight_sum += m.weight

    if weight_sum == 0:
        raise RuntimeError(f"{case_id}: no member contributed a prediction")
    pred = (prob_sum / weight_sum).argmax(0).astype(np.uint8)
    return pred, gt


def build_nets_for(
    members: Sequence[Member], split: str, fold_owner: Dict[str, int],
) -> Dict[str, Dict[str, List[Tuple[int, torch.nn.Module]]]]:
    """Map case -> member -> the networks that may predict it.

    On ``val`` only the fold that held the case out may predict it (that is what makes the
    number out-of-fold); on ``test`` every fold of every member is eligible.
    """
    if split == "test":
        allow_all = {m.name: m.nets for m in members}
        return {"*": allow_all}
    per_case: Dict[str, Dict[str, List]] = {}
    for case_id, fold in fold_owner.items():
        per_case[case_id] = {m.name: [(f, n) for f, n in m.nets if f == fold] for m in members}
    return per_case


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Ensemble several trained models (cross-experiment) and score them",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--member", nargs="+", action="append", required=True,
                    metavar=("CONFIG", "RUN_DIR"),
                    help="One model: its config YAML followed by its fold run dirs. "
                         "Repeat --member once per model.")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Per-member weights in --member order (default: equal). Folds inside "
                         "a member are always averaged equally.")
    ap.add_argument("--split", choices=["val", "test"], default="test",
                    help="val = out-of-fold CV of the ensemble; test = held-out set, all folds")
    ap.add_argument("--checkpoint", choices=["best", "final", "latest"], default="best")
    ap.add_argument("--output_dir", default=None,
                    help="Where to write CSV/JSON (default: <first run dir>/ensemble)")
    ap.add_argument("--preprocessed_dir", action="append", default=None,
                    help="Override each member's replica.preprocessed_dir, in --member order")
    ap.add_argument("--data_dir", default=None, help="Override data.train_dir for every member")
    ap.add_argument("--postprocess", action="store_true",
                    help="Apply the BraTS heuristic cleanup (not nnU-Net's)")
    ap.add_argument("--determine_postprocessing", action="store_true",
                    help="Determine nnU-Net post-processing on the ensemble's OOF predictions "
                         "(requires --split val) and write postprocessing.json")
    ap.add_argument("--postprocessing_json",
                    help="Apply previously determined nnU-Net ops (use on --split test)")
    ap.add_argument("--pp_criterion", choices=["labels", "regions"], default="labels",
                    help="Metric driving op selection: 'labels' = nnU-Net-faithful, "
                         "'regions' = the BraTS ET/TC/WT/RC score that is actually reported")
    ap.add_argument("--pp_min_gain", type=float, default=0.0,
                    help="Dice margin an op must clear. nnU-Net uses a strict > (0.0); raise "
                         "it (e.g. 1e-4) to demand a real gain rather than validation noise.")
    ap.add_argument("--no_tta", action="store_true", help="Disable mirroring TTA")
    ap.add_argument("--no_hd95", action="store_true", help="Skip HD95 (much faster)")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--save_predictions", default=None,
                    help="Directory to write the ensemble label maps as .npy (original volume "
                         "geometry). Implied by --determine_postprocessing.")
    args = ap.parse_args()

    specs = args.member
    bad = [s for s in specs if len(s) < 2]
    if bad:
        console.print("[red]Each --member needs a config YAML and at least one run dir[/red]")
        sys.exit(1)
    weights = args.weights or [1.0] * len(specs)
    if len(weights) != len(specs):
        console.print(f"[red]--weights has {len(weights)} values for {len(specs)} members[/red]")
        sys.exit(1)
    if any(w <= 0 for w in weights):
        console.print("[red]--weights must be positive[/red]")
        sys.exit(1)
    prep_overrides = args.preprocessed_dir or []
    if prep_overrides and len(prep_overrides) != len(specs):
        console.print(f"[red]--preprocessed_dir given {len(prep_overrides)} times for "
                      f"{len(specs)} members; give one per member or none[/red]")
        sys.exit(1)

    if args.determine_postprocessing and args.split != "val":
        console.print("[red]--determine_postprocessing requires --split val: determining on the "
                      "test set leaks it.[/red]")
        sys.exit(1)
    if args.determine_postprocessing and args.postprocessing_json:
        console.print("[red]--determine_postprocessing and --postprocessing_json are mutually "
                      "exclusive.[/red]")
        sys.exit(1)

    # Phase 1 — configs and splits only. Cheap, and it is where inconsistent members are
    # rejected, so a bad line-up costs seconds rather than a full round of checkpoint loading.
    members: List[Member] = []
    for i, spec in enumerate(specs):
        cfg_path, run_dirs = spec[0], spec[1:]
        try:
            members.append(describe_member(
                cfg_path, run_dirs, args.checkpoint, weights[i],
                preprocessed_dir=prep_overrides[i] if prep_overrides else None,
                data_dir=args.data_dir,
            ))
        except (FileNotFoundError, ValueError, KeyError) as exc:
            console.print(f"[red]{Path(cfg_path).stem}: {exc}[/red]")
            sys.exit(1)

    try:
        num_classes, regions, class_names = verify_members(members)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    print_members(members, args.split)

    # Phase 2 — the expensive part, now that the line-up is known to be coherent.
    for i, m in enumerate(members):
        console.print(f"[dim]loading {len(m.run_dirs)} checkpoint(s) for member "
                      f"{i + 1}/{len(members)}: {m.name}[/dim]")
        try:
            attach_networks(m, args.checkpoint, num_classes)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta = not args.no_tta
    mirror_axes = (0, 1, 2) if tta else None
    console.print(f"[dim]mirroring TTA: {'on (8 passes/tile)' if tta else 'off'}  |  "
                  f"device: {device}[/dim]")

    ref = members[0]
    if args.split == "val":
        fold_owner = {c: i for i, (_, va) in enumerate(ref.folds) for c in va}
        case_ids = [c for _, va in ref.folds for c in va]
    else:
        fold_owner = {}
        case_ids = list(ref.test_ids)
        if not case_ids:
            console.print("[red]No held-out test cases — data.kfold_holdout_test is false[/red]")
            sys.exit(1)
    if args.max_cases:
        case_ids = case_ids[: args.max_cases]

    # On val, every member should own every fold; otherwise the ensemble silently changes
    # composition from case to case as members drop out.
    if args.split == "val":
        n_folds = len(ref.folds)
        for m in members:
            gaps = sorted(set(range(n_folds)) - set(m.fold_ids))
            if gaps:
                console.print(f"[yellow]{m.name} is missing fold(s) {gaps}; it will not "
                              f"contribute to their validation cases, so the ensemble is a "
                              f"different mixture on those cases.[/yellow]")

    nets_map = build_nets_for(members, args.split, fold_owner)

    def nets_for_case(case_id: str) -> Dict[str, List]:
        return nets_map.get("*") or nets_map[case_id]

    if args.split == "val":
        empty = [c for c in case_ids if not any(nets_for_case(c).values())]
        if empty:
            console.print(f"[yellow]{len(empty)} validation case(s) have no matching fold "
                          f"checkpoint at all and will be skipped (e.g. {empty[:3]}).[/yellow]")
            case_ids = [c for c in case_ids if c not in set(empty)]
        if not case_ids:
            console.print("[red]No validation case has a matching fold checkpoint[/red]")
            sys.exit(1)

    # Open each member's cache over exactly the cases we will score: this validates up front
    # that every case was preprocessed for every member, and caches the pickled properties.
    for m in members:
        try:
            m.ds = PreprocessedDataset(m.dataset_folder, case_ids)
        except FileNotFoundError as exc:
            console.print(f"[red]{m.name}: {exc}[/red]")
            sys.exit(1)

    postprocess_kwargs = None
    if args.postprocess:
        from src.evaluation.postprocessing import postprocess_kwargs_from_config
        postprocess_kwargs = postprocess_kwargs_from_config(ref.config)
        console.print("[dim]heuristic BraTS post-processing: on (not nnU-Net's)[/dim]")

    nnunet_ops = None
    if args.postprocessing_json:
        with open(args.postprocessing_json) as f:
            nnunet_ops = json.load(f).get("operations", [])
        console.print(f"[cyan]nnU-Net post-processing ops: {nnunet_ops or 'none'}[/cyan]")

    out_dir = Path(args.output_dir) if args.output_dir else ref.run_dirs[0] / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(args.save_predictions) if args.save_predictions else (
        out_dir / f"predictions_{args.split}" if args.determine_postprocessing else None)
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]{len(case_ids)} case(s), {len(members)} member(s), "
                  f"{sum(len(m.nets) for m in members)} checkpoint(s) total[/bold]")

    rows = []
    for case_id in tqdm(case_ids, desc="ensemble"):
        pred, gt = ensemble_case(
            case_id, members, nets_for_case(case_id), num_classes, device, mirror_axes,
        )
        if pred_dir is not None:
            np.save(pred_dir / f"{case_id}.npy", pred)      # raw, before post-processing
        if nnunet_ops:
            from src.evaluation.nnunet_postprocessing import apply_postprocessing
            pred = apply_postprocessing(pred, nnunet_ops)
        if postprocess_kwargs is not None:
            from src.evaluation.postprocessing import postprocess_prediction
            pred = postprocess_prediction(pred, **postprocess_kwargs)
        rows.append(score_case(pred, gt, case_id, regions, class_names,
                               compute_hd=not args.no_hd95))

    df = pd.DataFrame(rows)
    title = (f"Ensemble of {len(members)} model(s) — "
             f"{'out-of-fold CV' if args.split == 'val' else 'held-out test'} "
             f"(checkpoint_{args.checkpoint})")
    summary = print_summary(df, regions, title)
    summary.update({
        "split": args.split, "checkpoint": args.checkpoint, "num_cases": len(df), "tta": tta,
        "heuristic_postprocess": postprocess_kwargs is not None,
        "nnunet_postprocess_ops": nnunet_ops,
        "members": [{"name": m.name, "weight": m.weight, "folds": sorted(m.fold_ids),
                     "input_channels": m.num_input_channels,
                     "run_dirs": [str(p) for p in m.run_dirs]} for m in members],
    })

    suffix = f"{args.split}_{args.checkpoint}"
    if postprocess_kwargs:
        suffix += "_pp"
    if nnunet_ops:
        suffix += "_nnpp"
    df.to_csv(out_dir / f"ensemble_metrics_{suffix}.csv", index=False)
    with open(out_dir / f"ensemble_summary_{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    console.print(f"[green]Wrote {out_dir}/ensemble_metrics_{suffix}.csv and "
                  f"ensemble_summary_{suffix}.json[/green]")

    if args.determine_postprocessing:
        def load_pair(case_id: str):
            """Cached ensemble prediction plus GT, both in original-volume geometry."""
            pred = np.load(pred_dir / f"{case_id}.npy")
            _, seg, props = ref.ds.load_case(case_id)
            return pred, uncrop_segmentation(seg[0], props)

        determine_and_report(
            case_ids, load_pair, out_dir, num_classes, regions, class_names,
            args.pp_criterion, compute_hd=not args.no_hd95,
            tag=f"ensemble_{args.split}_{args.checkpoint}", eps=args.pp_min_gain,
        )


if __name__ == "__main__":
    main()
