#!/usr/bin/env python3
"""Equivalence checks for post-processing and TTA: our code vs native nnU-Net.

Companion to ``equiv_checks.py`` (which covers training). Post-processing and inference-time
augmentation are where a plausible-looking reimplementation quietly diverges: scipy's
connected-component default is 6-neighbour where nnU-Net's is 26, a "largest component" that
breaks size ties deletes co-largest blobs, both-empty Dice scored as 1.0 instead of NaN
shifts every acceptance decision, and averaging softmax across flips is a different
estimator from averaging logits. Each of those is a silent Dice change, so each is asserted
here against the real implementation rather than argued about in a comment.

Run from brats-segmentation/:
    python3 tools/replica_parity/postprocessing_checks.py [check ...]

Checks needing the native tree (``components``, ``dice``, ``gaussian``, ``steps``) import
from the vendored 2.4.2 source under nnunet_data/, or from an installed nnunetv2 when the
symbol exists there. They skip with a message if neither is available; the rest always run.
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
NATIVE = REPO / "nnunet_data" / "replica_parity" / "nnUNet242"
sys.path.insert(0, str(REPO))
if NATIVE.is_dir():
    sys.path.insert(0, str(NATIVE))

from src.evaluation.nnunet_postprocessing import (  # noqa: E402
    _dice, determine_postprocessing, remove_all_but_largest, summarize,
)
from src.evaluation.postprocessing import _fill_holes, postprocess_prediction  # noqa: E402
from src.nnunet_replica.inference import (  # noqa: E402
    _maybe_mirror_and_predict, _mirror_axis_sets, compute_gaussian,
    compute_steps_for_sliding_window,
)

FG = [1, 2, 3, 4]
REGIONS = {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3], "RC": [4]}
PASS, FAIL, SKIP = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mSKIP\033[0m"


def _random_seg(rng, shape=(40, 40, 40), n_blobs=8):
    """Blobby multi-class volume with disconnected parts, equal-size blobs (tie stress),
    and single-voxel speckle (corner-touching stress for connectivity)."""
    seg = np.zeros(shape, dtype=np.uint8)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for _ in range(n_blobs):
        c = rng.integers(4, np.array(shape) - 4)
        r = rng.integers(2, 6)
        seg[(zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= r ** 2] = int(rng.integers(1, 5))
    for i in rng.integers(0, shape[0], size=(30, 3)):
        seg[tuple(i)] = int(rng.integers(1, 5))
    return seg


# ── checks ────────────────────────────────────────────────────────────────────

def check_components():
    """remove_all_but_largest == nnU-Net's remove_all_but_largest_component_from_segmentation."""
    try:
        from nnunetv2.postprocessing.remove_connected_components import (
            remove_all_but_largest_component_from_segmentation as nn_remove,
        )
    except ImportError:
        return None, "nnunetv2 not importable"

    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(30):
        seg = _random_seg(rng)
        for labels in ([1], [2], [3], [4], [1, 3], [1, 2, 3], FG):
            mine = remove_all_but_largest(seg, labels)
            theirs = nn_remove(seg, labels if len(labels) > 1 else labels[0])
            bad += not np.array_equal(mine, theirs)
    return bad == 0, f"30 volumes x 7 label sets, {bad} mismatches (26-connectivity, ties kept)"


def check_dice():
    """Per-label Dice matches nnU-Net's, including NaN when both masks are empty."""
    try:
        from nnunetv2.evaluation.evaluate_predictions import (
            compute_tp_fp_fn_tn, region_or_label_to_mask,
        )
    except ImportError:
        return None, "nnunetv2 not importable"

    rng = np.random.default_rng(1)
    bad = nan_cases = 0
    pairs = [(_random_seg(rng), _random_seg(rng)) for _ in range(20)]
    # Force the both-empty branch: label 4 appears in neither volume, and one pair is a
    # fully empty prediction against an empty reference.
    no4 = _random_seg(rng)
    no4[no4 == 4] = 2
    pairs.append((no4.copy(), no4.copy()))
    pairs.append((np.zeros((8, 8, 8), np.uint8), np.zeros((8, 8, 8), np.uint8)))

    for pred, gt in pairs:
        for labels in ([1], [2], [3], [4], [1, 3], [1, 2, 3]):
            mine = _dice(pred, gt, labels)
            r = labels[0] if len(labels) == 1 else tuple(labels)
            tp, fp, fn, _ = compute_tp_fp_fn_tn(
                region_or_label_to_mask(gt, r), region_or_label_to_mask(pred, r), None)
            theirs = np.nan if (tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
            if np.isnan(theirs):
                nan_cases += 1
            bad += not ((np.isnan(mine) and np.isnan(theirs)) or np.isclose(mine, theirs))
    n = len(pairs) * 6
    if nan_cases == 0:
        return False, "no both-empty case was exercised — the NaN convention went untested"
    return bad == 0, f"{n} comparisons, {bad} mismatches ({nan_cases} both-empty -> NaN)"


def check_aggregation():
    """nanmean per label, then mean over labels — not per-case-mean first."""
    per_case = [{"1": 0.8, "2": float("nan"), "3": 0.6},
                {"1": 0.4, "2": 0.5, "3": float("nan")}]
    s = summarize(per_case, ["1", "2", "3"])
    expect = {"1": 0.6, "2": 0.5, "3": 0.6}
    ok = (all(np.isclose(s["mean"][k], v) for k, v in expect.items())
          and np.isclose(s["foreground_mean"], np.mean(list(expect.values()))))
    naive = float(np.mean([np.nanmean(list(c.values())) for c in per_case]))
    return ok, (f"foreground_mean={s['foreground_mean']:.4f}; per-case-first would give "
                f"{naive:.4f} (differs by {abs(naive - s['foreground_mean']):.4f})")


def check_defensive_guard():
    """The merged-foreground op is rejected when it degrades any single label.

    nnU-Net accepts keep-largest-over-all-foreground only if the mean improves AND no
    individual label gets worse. Here the op raises the mean but deletes a true, spatially
    separate label-4 lesion, so it must be rejected.
    """
    gt = np.zeros((40, 40, 40), np.uint8)
    gt[5:20, 5:20, 5:20] = 1
    gt[30:34, 30:34, 30:34] = 4          # true lesion, disconnected from the main blob
    pred = gt.copy()
    pred[25:27, 5:7, 5:7] = 2            # false-positive island

    ops, report = determine_postprocessing([pred], [gt], FG, REGIONS)
    merged = report["log"][0]
    ok = merged["accepted"] is False and [1, 2, 3, 4] not in ops
    return ok, (f"merged-fg {merged['judged_value_before']} -> {merged['judged_value']}, "
                f"accepted={merged['accepted']}; ops={ops}")


def check_no_leakage_shape():
    """Determination reports ops as label sets that apply_postprocessing can consume."""
    from src.evaluation.nnunet_postprocessing import apply_postprocessing

    rng = np.random.default_rng(3)
    gt = np.zeros((40, 40, 40), np.uint8)
    zz, yy, xx = np.ogrid[:40, :40, :40]
    gt[(zz - 20) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2 <= 64] = 3
    preds, gts = [], []
    for _ in range(6):
        p = gt.copy()
        c = rng.integers(2, 8, size=3)
        p[(zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= 4] = 3   # spurious ET
        preds.append(p)
        gts.append(gt)
    ops, _ = determine_postprocessing(preds, gts, FG, REGIONS)
    before = int((preds[0] == 3).sum())
    after = int((apply_postprocessing(preds[0], ops) == 3).sum())
    return (bool(ops) and after < before), f"ops={ops}; ET voxels {before} -> {after}"


def check_heuristic_holes():
    """The heuristic cleanup must not punch holes through WT/TC, and fills holes in 3D."""
    kw = dict(foreground_classes=(1, 2, 3, 4), et_label=3, et_fold_into=1,
              wt_labels=(1, 2, 3), tc_labels=(1, 3), fill_into=1,
              et_min_voxels=0, min_component_size=50)
    pred = np.zeros((40, 40, 40), np.uint8)
    pred[10:30, 10:30, 10:30] = 2
    pred[15:25, 15:25, 15:25] = 3
    pred[18:20, 18:20, 18:23] = 1        # 20-voxel interior speckle, below threshold
    pred[35:37, 35:37, 35:38] = 1        # 12-voxel isolated island = real noise

    wt0 = int(np.isin(pred, [1, 2, 3]).sum())
    legacy = postprocess_prediction(pred, fill_holes=False, small_component_mode="always", **kw)
    fixed = postprocess_prediction(pred, fill_holes=False, small_component_mode="reassign", **kw)
    lost_legacy = wt0 - int(np.isin(legacy, [1, 2, 3]).sum())
    lost_fixed = wt0 - int(np.isin(fixed, [1, 2, 3]).sum())

    # open channel is not a 3D hole; an enclosed cavity is
    cup = np.zeros((20, 20, 20), bool)
    cup[5:15, 5:15, 5:15] = True
    cup[8:12, 8:12, 8:20] = False
    box = np.zeros((20, 20, 20), bool)
    box[5:15, 5:15, 5:15] = True
    box[9:11, 9:11, 9:11] = False

    invented_axial = int((_fill_holes(cup, True) & ~cup).sum())
    ok = (lost_fixed == 12 and lost_legacy == 32
          and (fixed[18:20, 18:20, 18:23] == 3).all()
          and (fixed[35:37, 35:37, 35:38] == 0).all()
          and (_fill_holes(cup, False) == cup).all()
          and (_fill_holes(box, False) & ~box).sum() == 8)
    return ok, (f"WT lost: legacy {lost_legacy} vx (speckle+island) vs reassign {lost_fixed} vx "
                f"(island only); axial fill invents {invented_axial} vx on an open channel")


def check_tta():
    """Mirroring TTA is all 8 flips with logits averaged, matching nnU-Net exactly."""
    class Asymmetric(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.conv = nn.Conv3d(4, 5, 3, padding=1)
            self.register_buffer("ramp", torch.linspace(0, 1, 16).view(1, 1, 16, 1, 1))

        def forward(self, x):
            return self.conv(x) * (1 + self.ramp) + x[:, :1] * 0.3

    def reference(network, x, mirror_axes):
        """Verbatim nnUNetPredictor._internal_maybe_mirror_and_predict (v2.4.2)."""
        prediction = network(x)
        if mirror_axes is not None:
            axes = [m + 2 for m in mirror_axes]
            combos = [c for i in range(len(axes)) for c in itertools.combinations(axes, i + 1)]
            for a in combos:
                prediction += torch.flip(network(torch.flip(x, a)), a)
            prediction /= (len(combos) + 1)
        return prediction

    net = Asymmetric().eval()
    torch.manual_seed(1)
    x = torch.randn(1, 4, 16, 16, 16)
    if len(_mirror_axis_sets([0, 1, 2])) != 8:
        return False, "expected 8 flip combinations for 3 axes"

    worst = 0.0
    with torch.no_grad():
        for axes in [(0, 1, 2), (0, 1), (0,), None]:
            mine = _maybe_mirror_and_predict(net, x.clone(), axes)
            theirs = reference(net, x.clone(), list(axes) if axes else None)
            worst = max(worst, (mine - theirs).abs().max().item())
        # control: the net must actually be flip-sensitive, else the test is vacuous
        sensitivity = (net(x) - torch.flip(net(torch.flip(x, (2,))), (2,))).abs().max().item()
        # and logit vs softmax averaging must genuinely differ
        avg_logits = _maybe_mirror_and_predict(net, x.clone(), (0, 1, 2))
        softmax_first = sum(
            torch.flip(torch.softmax(net(torch.flip(x, tuple(a + 2 for a in s))), 1),
                       tuple(a + 2 for a in s)) if s else torch.softmax(net(x), 1)
            for s in _mirror_axis_sets([0, 1, 2])
        ) / 8
    gap = (torch.softmax(avg_logits, 1) - softmax_first).abs().max().item()
    ok = worst < 1e-6 and sensitivity > 0.01 and gap > 1e-5
    return ok, (f"8 flips, max|d| vs native = {worst:.2e}; net flip-sensitivity "
                f"{sensitivity:.2f}; logit- vs softmax-averaging gap {gap:.4f}")


def check_gaussian():
    """Tile weighting matches nnU-Net's compute_gaussian."""
    try:
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian as nn_g
    except ImportError:
        return None, "native compute_gaussian not importable"
    cpu = torch.device("cpu")
    worst = 0.0
    for patch in [(128, 128, 128), (112, 128, 128), (96, 96, 96), (80, 96, 112)]:
        a = compute_gaussian(patch).cpu().numpy()
        b = nn_g(patch, dtype=torch.float32, device=cpu).cpu().numpy()
        worst = max(worst, float(np.abs(a - b).max()))
        if a.min() <= 0:
            return False, "gaussian has non-positive weights — will divide by zero"
    return worst < 1e-6, f"4 patch sizes, max|d| = {worst:.2e}, all weights > 0"


def check_steps():
    """Sliding-window step placement matches nnU-Net's."""
    try:
        from nnunetv2.inference.sliding_window_prediction import (
            compute_steps_for_sliding_window as nn_s,
        )
    except ImportError:
        return None, "native compute_steps_for_sliding_window not importable"
    n = bad = 0
    for img in [(160, 180, 140), (240, 240, 155), (128, 128, 128), (200, 150, 130)]:
        for patch in [(128, 128, 128), (112, 128, 128)]:
            if any(i < p for i, p in zip(img, patch)):
                continue
            n += 1
            bad += compute_steps_for_sliding_window(img, patch, 0.5) != nn_s(img, patch, 0.5)
    return bad == 0, f"{n} image/patch combinations, {bad} mismatches"


CHECKS = {
    "components": check_components,
    "dice": check_dice,
    "aggregation": check_aggregation,
    "guard": check_defensive_guard,
    "determine": check_no_leakage_shape,
    "heuristic": check_heuristic_holes,
    "tta": check_tta,
    "gaussian": check_gaussian,
    "steps": check_steps,
}


def main():
    wanted = sys.argv[1:] or list(CHECKS)
    unknown = [w for w in wanted if w not in CHECKS]
    if unknown:
        print(f"unknown check(s): {unknown}\navailable: {', '.join(CHECKS)}")
        return 2

    failed = 0
    for name in wanted:
        try:
            ok, detail = CHECKS[name]()
        except Exception as exc:  # a check that cannot run is a failure, not a pass
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        if ok is None:
            print(f"{SKIP} {name:12s} {detail}")
            continue
        print(f"{PASS if ok else FAIL} {name:12s} {detail}")
        failed += not ok

    print(f"\n{len(wanted) - failed}/{len(wanted)} checks passed"
          if failed else f"\nall {len(wanted)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
