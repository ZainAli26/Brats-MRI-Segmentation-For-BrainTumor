#!/usr/bin/env python3
"""Checks for evaluate_ensemble.py — cross-model ensembling geometry, weighting and guards.

The ensemble averages models that may have been preprocessed with *different* input channel
counts, which means different nonzero-crop bounding boxes and therefore predictions that are
not voxel-aligned with each other. Everything here defends that seam:

* un-cropping to the original volume is lossless inside the crop and background-certain
  outside it (a flat-zero fill would instead let the loosest-cropping member decide the
  argmax on its own in the region only it covers);
* ``-1`` (outside-brain) in the cached ground truth becomes background rather than
  wrapping to 255 through uint8;
* member weights actually steer a disagreement, and folds inside a member stay equal;
* the consistency guards refuse a line-up whose members do not share a patient split —
  the failure that would silently score a model on data it trained on.

Run from brats-segmentation/:
    python3 tools/replica_parity/ensemble_checks.py [check ...]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import evaluate_ensemble as ee  # noqa: E402
from evaluate_ensemble import (  # noqa: E402
    Member, build_nets_for, ensemble_case, uncrop_probabilities, uncrop_segmentation,
    verify_members,
)

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"

C = 5
SHAPE = (10, 10, 8)
BBOX = [[2, 8], [3, 7], [1, 6]]
PROPS = {"shape_before_cropping": list(SHAPE), "bbox_used_for_cropping": BBOX}
CROPPED = tuple(hi - lo for lo, hi in BBOX)
REGIONS = {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3], "RC": [4]}
NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}


def _outside_mask():
    m = np.ones(SHAPE, bool)
    m[2:8, 3:7, 1:6] = False
    return m


# ── geometry ──────────────────────────────────────────────────────────────────

def check_uncrop_probs():
    """Un-cropping preserves the crop exactly and is background-certain outside it."""
    probs = np.zeros((C, *CROPPED), dtype=np.float32)
    probs[3] = 1.0
    full = uncrop_probabilities(probs, PROPS)
    outside = _outside_mask()
    ok = (full.shape == (C, *SHAPE)
          and np.allclose(full[(slice(None), slice(2, 8), slice(3, 7), slice(1, 6))], probs)
          and np.allclose(full.sum(0), 1.0)
          and np.allclose(full[0][outside], 1.0)
          and np.allclose(full[1:][:, outside], 0.0)
          and int(full.argmax(0)[outside].max()) == 0)
    return ok, (f"shape {full.shape}, channels sum to 1 everywhere, "
                f"{int(outside.sum())} outside-crop voxels are background-certain")


def check_differing_crops():
    """A member's opinion stands where only it has coverage, and blends where both do."""
    props_a = {"shape_before_cropping": list(SHAPE),
               "bbox_used_for_cropping": [[4, 6], [4, 6], [2, 4]]}     # narrow
    props_b = {"shape_before_cropping": list(SHAPE),
               "bbox_used_for_cropping": [[0, 10], [0, 10], [0, 8]]}   # full
    pa = np.zeros((C, 2, 2, 2), np.float32)
    pa[1] = 1.0                                   # A: confidently class 1, small box
    pb = np.zeros((C, *SHAPE), np.float32)
    pb[0], pb[2] = 0.6, 0.4                       # B: background-leaning, everywhere
    avg = (uncrop_probabilities(pa, props_a) + uncrop_probabilities(pb, props_b)) / 2
    b_only = int(avg[:, 0, 0, 0].argmax())
    overlap = int(avg[:, 4, 4, 2].argmax())
    ok = b_only == 0 and overlap == 1
    return ok, (f"B-only voxel -> class {b_only} (B's own call, undiluted); "
                f"overlap voxel -> class {overlap} (A's confidence wins)")


def check_uncrop_seg():
    """-1 (outside brain) becomes background, not 255."""
    seg = np.zeros(CROPPED, np.int8)
    seg[0, 0, 0] = 3
    seg[1, 1, 1] = -1
    gt = uncrop_segmentation(seg, PROPS)
    ok = (gt.shape == SHAPE and gt.dtype == np.uint8 and gt[2, 3, 1] == 3
          and gt[3, 4, 2] == 0 and int(gt.max()) == 3
          and int(gt[_outside_mask()].sum()) == 0)
    return ok, f"label at bbox origin, -1 -> 0, max label {int(gt.max())} (no uint8 wraparound)"


# ── weighting ─────────────────────────────────────────────────────────────────

class _FakeNet:
    """Returns a fixed one-hot logit volume; stands in for a trained network."""

    def __init__(self, cls: int):
        self.cls = cls

    def to(self, device):
        return self


def _fake_member(name: str, cls: int, weight: float) -> Member:
    class _RC:
        tile_step_size = 0.5
        amp = False

    class _DS:
        def load_properties(self, cid):
            return PROPS

        def load_case(self, cid):
            return (np.zeros((1, *CROPPED), np.float32),
                    np.zeros((1, *CROPPED), np.int8), PROPS)

    m = Member(name=name, config={}, rcfg=_RC(), plan_cfg=None, patch_size=[4, 4, 4],
               num_input_channels=1, dataset_folder=Path("-"), run_dirs=[], folds=[],
               test_ids=[], weight=weight)
    m.nets = [(0, _FakeNet(cls))]
    m.ds = _DS()
    return m


def _fake_predict(net, x, patch, num_classes, **kw):
    logits = torch.full((num_classes, *tuple(x.shape[1:])), -20.0)
    logits[net.cls] = 20.0
    return logits


def check_weighting():
    """Member weights steer a disagreement; equal weights leave it a tie."""
    orig = ee.predict_sliding_window
    ee.predict_sliding_window = _fake_predict
    results = []
    try:
        for wa, wb, expect in [(3.0, 1.0, 1), (1.0, 3.0, 2), (1.0, 1.0, None)]:
            ma, mb = _fake_member("A", 1, wa), _fake_member("B", 2, wb)
            pred, _ = ensemble_case("c", [ma, mb], {"A": ma.nets, "B": mb.nets},
                                    C, torch.device("cpu"), None)
            got = int(pred[4, 4, 3])
            results.append((wa, wb, got, expect))
    finally:
        ee.predict_sliding_window = orig
    ok = all(got == exp for _, _, got, exp in results if exp is not None)
    ok = ok and results[-1][2] in (1, 2)
    detail = "; ".join(f"{wa:g}:{wb:g} -> {got}" for wa, wb, got, _ in results)
    return ok, detail + " (last is an exact tie)"


def check_fold_equal_within_member():
    """Folds inside one member are averaged equally regardless of how many there are.

    Two folds voting class 1 and one voting class 2 must land on class 1, and the member's
    own weight must not scale with its fold count.
    """
    orig = ee.predict_sliding_window
    ee.predict_sliding_window = _fake_predict
    try:
        # Member A: 3 folds, 2 for class 1 and 1 for class 2 -> A's mean favours class 1.
        ma = _fake_member("A", 1, 1.0)
        ma.nets = [(0, _FakeNet(1)), (1, _FakeNet(1)), (2, _FakeNet(2))]
        pred, _ = ensemble_case("c", [ma], {"A": ma.nets}, C, torch.device("cpu"), None)
        majority = int(pred[4, 4, 3])

        # A (3 folds, all class 1) vs B (1 fold, class 2) at equal weight -> exact tie,
        # proving fold count does not inflate a member's influence.
        ma2 = _fake_member("A", 1, 1.0)
        ma2.nets = [(0, _FakeNet(1)), (1, _FakeNet(1)), (2, _FakeNet(1))]
        mb2 = _fake_member("B", 2, 1.0)
        pred2, _ = ensemble_case("c", [ma2, mb2], {"A": ma2.nets, "B": mb2.nets},
                                 C, torch.device("cpu"), None)
        tied = int(pred2[4, 4, 3])
    finally:
        ee.predict_sliding_window = orig
    ok = majority == 1 and tied in (1, 2)
    return ok, (f"2-of-3 folds for class 1 -> {majority}; 3 folds vs 1 fold at equal "
                f"weight -> {tied} (a tie, so fold count did not inflate A)")


# ── guards ────────────────────────────────────────────────────────────────────

def _member(name, *, classes=5, regions=None, names=None, val=None, test=None, train=None):
    folds = [(train or ["t1", "t2"], v) for v in (val or [["a", "b"], ["c", "d"]])]
    return Member(
        name=name,
        config={"data": {"num_classes": classes, "class_names": names or NAMES,
                         "modalities": ["t1c"] * 4},
                "evaluation": {"regions": regions or REGIONS}},
        rcfg=None, plan_cfg=None, patch_size=[32] * 3, num_input_channels=4,
        dataset_folder=Path("-"), run_dirs=[], folds=folds,
        test_ids=test if test is not None else ["z1", "z2"],
    )


def check_guards():
    """verify_members accepts a coherent line-up and names the specific inconsistency."""
    ref = _member("ref")
    cases = [
        ("num_classes", _member("m", classes=4), "label schemes"),
        ("regions", _member("m", regions={"ET": [3]}), "different regions"),
        ("class_names", _member("m", names={1: "X", 2: "SNFH", 3: "ET", 4: "RC"}),
         "different class names"),
        ("val split", _member("m", val=[["a", "X"], ["c", "d"]]), "validation set differs"),
        ("test set", _member("m", test=["z1", "z9"]), "different held-out test set"),
    ]
    failures = []
    for label, bad, expect in cases:
        try:
            verify_members([ref, bad])
            failures.append(f"{label} not rejected")
        except ValueError as exc:
            if expect not in str(exc):
                failures.append(f"{label} rejected but message lacked {expect!r}")

    # Differing TRAIN sets must be allowed: exp27 adds synthetic cases to train only.
    try:
        verify_members([ref, _member("m", train=["t1", "t2", "synthetic-1"])])
    except ValueError as exc:
        failures.append(f"differing train sets wrongly rejected: {exc}")

    # A coherent pair must pass and report the reference scheme.
    try:
        classes, regions, names = verify_members([ref, _member("m")])
        if (classes, regions, names) != (5, REGIONS, NAMES):
            failures.append(f"returned scheme wrong: {classes}, {regions}, {names}")
    except ValueError as exc:
        failures.append(f"coherent pair rejected: {exc}")

    return not failures, ("; ".join(failures) if failures else
                          f"{len(cases)} inconsistencies rejected, differing train sets "
                          f"allowed, coherent pair accepted")


def check_fold_routing():
    """On val only the owning fold may predict a case; on test every fold is eligible."""
    ma, mb = _fake_member("A", 1, 1.0), _fake_member("B", 2, 1.0)
    ma.nets = [(0, _FakeNet(1)), (1, _FakeNet(1))]
    mb.nets = [(0, _FakeNet(2)), (1, _FakeNet(2))]
    members = [ma, mb]

    owner = {"a": 0, "b": 0, "c": 1, "d": 1}
    val_map = build_nets_for(members, "val", owner)
    routed = {c: {n: [f for f, _ in nets] for n, nets in per.items()}
              for c, per in val_map.items()}
    val_ok = all(routed[c] == {"A": [owner[c]], "B": [owner[c]]} for c in owner)

    test_map = build_nets_for(members, "test", {})
    test_ok = ("*" in test_map
               and [f for f, _ in test_map["*"]["A"]] == [0, 1]
               and [f for f, _ in test_map["*"]["B"]] == [0, 1])
    return val_ok and test_ok, (f"val: case->folds {routed['a']} for a (fold 0), "
                                f"{routed['c']} for c (fold 1); test: all folds eligible")


CHECKS = {
    "uncrop_probs": check_uncrop_probs,
    "differing_crops": check_differing_crops,
    "uncrop_seg": check_uncrop_seg,
    "weighting": check_weighting,
    "fold_equal": check_fold_equal_within_member,
    "guards": check_guards,
    "fold_routing": check_fold_routing,
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
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"{PASS if ok else FAIL} {name:16s} {detail}")
        failed += not ok

    print(f"\nall {len(wanted)} checks passed" if not failed
          else f"\n{len(wanted) - failed}/{len(wanted)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
