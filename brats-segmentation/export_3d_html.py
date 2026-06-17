#!/usr/bin/env python3
"""BraTS — Export interactive side-by-side 3-D segmentation maps (HTML).

For every case, writes ONE self-contained interactive HTML file containing two
camera-synced 3-D scenes:

    ┌────────────── Ground Truth ──────────────┬────────────── Prediction ──────────────┐
    │   marching-cubes surfaces of the          │   marching-cubes surfaces of the model  │
    │   annotated tumour (NCR / ED / ET)        │   prediction (NCR / ED / ET)            │
    └───────────────────────────────────────────┴─────────────────────────────────────────┘

Rotating/zooming one scene drives the other (synchronised camera) so the
annotation and the prediction can be compared in parallel.

Output layout
-------------
    <output_dir>/
        test_set/
            BraTS-GLI-00123-000.html
            plotly.min.js          (written once, shared by all files in the dir)
        train_sample/
            BraTS-GLI-00045-000.html
            plotly.min.js

Test-set source
---------------
The exp18 fold models are trained with k-fold CV over *all* data, so the
patient-level "test" split (create_patient_splits) is NOT held out for a single
fold model.  Default `--test_source val_fold0` uses the fold's genuinely-unseen
validation cases.  Pass `--test_source patient` to instead use the 10% patient
hold-out (matches visualize_segmentations.py's `test` split).

Usage
-----
    python3 export_3d_html.py \\
        --fold_dirs runs/nnunet_v2_20260507_021354_fold0 \\
        --config experiments/exp18_nnunet_v2_residual_5fold.yaml \\
        --output_dir visualizations_3d \\
        --n_train 50

    # quick smoke test — 2 cases per split
    python3 export_3d_html.py ... --max_cases 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn,
)
from scipy import ndimage
from torch.cuda.amp import autocast

from src.data.dataset import build_file_list
from src.data.preprocessing import get_val_transforms
from src.data.splits import create_kfold_splits, create_patient_splits
from src.evaluation.postprocessing import postprocess_prediction
from src.models.factory import create_model
from src.utils import inference_wrapper
from src.utils.experiment import load_config

console = Console()

# ── colours (match visualize_segmentations.py) ────────────────────────────────
DARK_BG = "#0d1117"
TEXT_CLR = "#e6edf3"
HEX_COLORS = {1: "#FF4545", 2: "#FFD700", 3: "#00CFFF"}   # NCR, ED, ET
LABEL_NAMES = {1: "NCR (necrotic core)", 2: "ED (oedema)", 3: "ET (enhancing)"}

# JS injected into every HTML: keep both 3-D cameras in lock-step.
_CAMERA_SYNC_JS = """
var gd = document.getElementById('{plot_id}');
var _syncing = false;
gd.on('plotly_relayout', function(ed) {
    if (_syncing) return;
    var cam = ed['scene.camera'] || ed['scene2.camera'];
    if (!cam) return;
    _syncing = true;
    Plotly.relayout(gd, {'scene.camera': cam, 'scene2.camera': cam})
          .then(function() { _syncing = false; });
});
"""


# ── geometry helpers ───────────────────────────────────────────────────────────

def _tumor_center(seg: np.ndarray) -> tuple:
    fg = seg > 0
    if fg.any():
        com = ndimage.center_of_mass(fg)
        return tuple(min(int(round(c)), seg.shape[i] - 1) for i, c in enumerate(com))
    return tuple(s // 2 for s in seg.shape)


def _downsample_seg(seg: np.ndarray, target_max: int) -> np.ndarray:
    """Nearest-neighbour downsample a label volume so its largest dim ≈ target_max."""
    from skimage.transform import resize as sk_resize
    h, w, d = seg.shape
    scale = target_max / max(h, w, d)
    if scale >= 1.0:
        return seg.astype(np.int32)
    new_shape = tuple(max(int(round(s * scale)), 4) for s in seg.shape)
    small = sk_resize(seg.astype(np.float32), new_shape,
                      order=0, anti_aliasing=False, preserve_range=True)
    return np.round(small).astype(np.int32)


def _class_meshes(seg: np.ndarray, min_voxels: int = 8):
    """Yield (class, verts, faces) marching-cubes meshes for each tumour class."""
    from skimage.measure import marching_cubes
    for cls in (2, 1, 3):  # ED (largest) first, ET last
        mask = (seg == cls).astype(np.float32)
        if mask.sum() < min_voxels:
            continue
        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5, allow_degenerate=False)
        except Exception:
            continue
        yield cls, verts, faces


def _case_dice(pred: np.ndarray, lab: np.ndarray, regions: dict) -> dict:
    """Per-region Dice for a single case (regions map name -> [label indices])."""
    out = {}
    for rname, indices in regions.items():
        pred_r = np.isin(pred, indices)
        lab_r = np.isin(lab, indices)
        inter = np.logical_and(pred_r, lab_r).sum()
        union = pred_r.sum() + lab_r.sum()
        out[rname] = float(2.0 * inter / (union + 1e-7))
    return out


# ── figure building ─────────────────────────────────────────────────────────────

def _add_meshes(fig, seg: np.ndarray, scene: str, col: int):
    """Add one Mesh3d trace per tumour class to the given subplot scene."""
    import plotly.graph_objects as go
    for cls, verts, faces in _class_meshes(seg):
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=HEX_COLORS[cls],
                opacity=0.55 if cls == 2 else 0.9,   # oedema translucent so core shows through
                name=LABEL_NAMES[cls],
                showlegend=(col == 1),               # one legend entry per class
                legendgroup=str(cls),
                hoverinfo="name",
                flatshading=True,
                lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15),
            ),
            row=1, col=col,
        )


def _build_figure(case_id, label_np, pred_np, dice, downsample):
    """Return a plotly Figure with GT (left) and prediction (right) 3-D scenes."""
    from plotly.subplots import make_subplots

    gt_small = _downsample_seg(label_np, downsample)
    pr_small = _downsample_seg(pred_np, downsample)

    region_str = "  |  ".join(f"{r}: {v:.3f}" for r, v in dice.items())
    mean_dice = float(np.mean(list(dice.values()))) if dice else 0.0

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Ground Truth (annotation)", "Prediction (model)"),
        horizontal_spacing=0.02,
    )

    _add_meshes(fig, gt_small, "scene", col=1)
    _add_meshes(fig, pr_small, "scene2", col=2)

    scene_kw = dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode="data",
        bgcolor=DARK_BG,
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
    )
    fig.update_layout(
        title=dict(
            text=f"<b>{case_id}</b><br><span style='font-size:13px'>{region_str}  |  Mean: {mean_dice:.3f}</span>",
            x=0.5, xanchor="center", font=dict(color=TEXT_CLR, size=18),
        ),
        scene=scene_kw, scene2=scene_kw,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                    x=0.5, xanchor="center", orientation="h", y=-0.02),
        margin=dict(l=0, r=0, t=70, b=10),
    )
    return fig


# ── inference + export loop ───────────────────────────────────────────────────

def process_cases(models, dataloader, config, device, output_dir,
                  use_postproc, postproc_kwargs, downsample, max_cases):
    import plotly.io as pio

    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_size = config["preprocessing"]["spatial_size"]
    sw_batch = config["training"]["sw_batch_size"]
    sw_overlap = config["training"]["sw_overlap"]
    num_classes = config["data"]["num_classes"]
    regions = config["evaluation"]["regions"]
    use_amp = config["training"]["amp"] and device.type == "cuda"

    post_pred_d = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    n_done = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Exporting", total=min(len(dataloader), max_cases or len(dataloader)))

        for batch_data in dataloader:
            if max_cases is not None and n_done >= max_cases:
                break

            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            case_id = batch_data.get("case_id", ["unknown"])[0]

            with torch.no_grad():
                prob_acc = None
                for m in models:
                    with autocast(enabled=use_amp):
                        logits = sliding_window_inference(
                            images, spatial_size, sw_batch,
                            inference_wrapper(m), overlap=sw_overlap,
                        )
                    prob = torch.softmax(logits, dim=1)
                    prob_acc = prob if prob_acc is None else prob_acc + prob
                ensemble_prob = prob_acc / len(models)

            pred_oh = post_pred_d(decollate_batch(ensemble_prob)[0])
            lab_oh = post_label(decollate_batch(labels)[0])
            pred_np = pred_oh.argmax(dim=0).cpu().numpy().astype(np.int32)
            lab_np = lab_oh.argmax(dim=0).cpu().numpy().astype(np.int32)

            if use_postproc:
                pred_np = postprocess_prediction(pred_np, **postproc_kwargs)

            dice = _case_dice(pred_np, lab_np, regions)
            fig = _build_figure(case_id, lab_np, pred_np, dice, downsample)

            # include_plotlyjs="directory" writes plotly.min.js into the dir once
            # (on the first file) and references it from every HTML — keeps each
            # file small while staying fully offline-usable.
            pio.write_html(
                fig, file=str(output_dir / f"{case_id}.html"),
                include_plotlyjs="directory", full_html=True,
                post_script=_CAMERA_SYNC_JS,
                config={"displaylogo": False},
            )

            n_done += 1
            prog.update(task, advance=1)

    console.print(f"[bold green]Saved {n_done} HTML files → {output_dir}[/bold green]")


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Export side-by-side 3-D GT/prediction HTML per case")
    p.add_argument("--fold_dirs", nargs="+", required=True,
                   help="Fold run directories containing best_model.pth (multiple → ensemble)")
    p.add_argument("--config", required=True, help="Experiment config YAML")
    p.add_argument("--output_dir", default="./visualizations_3d",
                   help="Base output directory (default: ./visualizations_3d)")
    p.add_argument("--test_source", choices=["val_fold0", "val_fold1", "val_fold2",
                                             "val_fold3", "val_fold4", "patient"],
                   default="val_fold0",
                   help="What the test_set/ export uses. val_foldN = fold's unseen val cases "
                        "(honest for that fold model). 'patient' = 10%% patient hold-out "
                        "(may overlap k-fold training data). Default: val_fold0")
    p.add_argument("--n_train", type=int, default=50,
                   help="Number of training cases to also export (default: 50; 0 to skip)")
    p.add_argument("--downsample", type=int, default=96,
                   help="Max voxel dim for marching-cubes mesh (lower = lighter HTML). Default: 96")
    p.add_argument("--no_postproc", action="store_true", help="Export raw predictions (no post-proc)")
    p.add_argument("--et_min_voxels", type=int, default=250)
    p.add_argument("--min_component_size", type=int, default=50)
    p.add_argument("--max_cases", type=int, default=None,
                   help="Cap cases per split (smoke test)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--data_dir", type=str, default=None,
                   help="Override data.train_dir from config (useful inside Docker)")
    args = p.parse_args()

    try:
        import plotly  # noqa: F401
    except ImportError:
        console.print("[red bold]plotly is not installed.[/red bold] "
                      "Run inside Docker (docker-compose run --rm export-3d-test) "
                      "or `pip install plotly`.")
        sys.exit(1)

    config = load_config(args.config)
    if args.data_dir:
        config["data"]["train_dir"] = args.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    postproc_kwargs = dict(
        et_min_voxels=args.et_min_voxels,
        min_component_size=args.min_component_size,
        fill_holes=True,
    )

    data_dir = Path(config["data"]["train_dir"]).expanduser()
    n_folds = config["data"].get("n_folds", 5)
    modalities = config["data"]["modalities"]
    label_map = {int(k): int(v) for k, v in config["data"]["label_map"].items()}
    spatial_size = config["preprocessing"]["spatial_size"]
    base_out = Path(args.output_dir)

    console.print(Panel.fit(
        f"[bold cyan]BraTS — 3-D GT vs Prediction Export[/bold cyan]\n"
        f"[dim]Test source: {args.test_source} | Train sample: {args.n_train} | "
        f"Post-proc: {not args.no_postproc} | Downsample: {args.downsample} | "
        f"Device: {device}[/dim]",
        border_style="bright_blue",
    ))

    if not data_dir.exists():
        console.print(f"[red]Data dir not found: {data_dir}[/red]")
        sys.exit(1)

    # ── resolve test + train case lists ──────────────────────────────────────
    kfold_splits = create_kfold_splits(str(data_dir), n_folds=n_folds,
                                       seed=config["data"]["split_seed"])
    train_cases_master, _, patient_test = create_patient_splits(
        str(data_dir), split_ratios=[0.75, 0.15, 0.10], seed=config["data"]["split_seed"],
    )

    if args.test_source == "patient":
        test_cases = patient_test
        console.print("[yellow]Note: patient test split may overlap k-fold training "
                      "data for single-fold models.[/yellow]")
    else:
        fold_idx = int(args.test_source.split("fold")[-1])
        _, test_cases = kfold_splits[fold_idx]

    train_cases = train_cases_master[:args.n_train] if args.n_train > 0 else []

    if args.max_cases:
        test_cases = test_cases[:args.max_cases]
        train_cases = train_cases[:args.max_cases]

    # ── load models once ──────────────────────────────────────────────────────
    models = []
    for fold_dir in [Path(d).expanduser() for d in args.fold_dirs]:
        ckpt_path = fold_dir / "best_model.pth"
        if not ckpt_path.exists():
            console.print(f"[yellow]No checkpoint in {fold_dir}, skipping[/yellow]")
            continue
        m = create_model(config)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device).eval()
        console.print(f"  [green]Loaded {fold_dir.name}[/green] — epoch {ckpt.get('epoch', '?')}")
        models.append(m)

    if not models:
        console.print("[red bold]No valid checkpoints found.[/red bold]")
        sys.exit(1)

    val_transform = get_val_transforms(spatial_size, modalities, label_map)

    jobs = [("test_set", test_cases)]
    if train_cases:
        jobs.append(("train_sample", train_cases))

    for split_label, cases in jobs:
        console.print(f"\n[bold]Exporting [cyan]{split_label}[/cyan] "
                      f"({len(cases)} cases) → [dim]{base_out / split_label}[/dim][/bold]")
        file_list = build_file_list(cases, modalities, include_label=True)
        if not file_list:
            console.print(f"[yellow]No labelled cases found for {split_label}[/yellow]")
            continue
        ds = Dataset(file_list, transform=val_transform)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        process_cases(
            models, loader, config, device, base_out / split_label,
            use_postproc=not args.no_postproc, postproc_kwargs=postproc_kwargs,
            downsample=args.downsample, max_cases=args.max_cases,
        )

    console.print(f"\n[bold green]Done. Open the .html files in any browser → {base_out}[/bold green]")


if __name__ == "__main__":
    main()
