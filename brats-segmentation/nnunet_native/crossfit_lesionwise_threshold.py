#!/usr/bin/env python3
"""Cross-fitted validation of the small-component threshold chosen by the sweep.

The sweep picks the best N on the same out-of-fold cases it reports, so its winner is
in-sample. This script measures how much that selection actually inflates the number:
for each fold k, choose N on the other four folds' cases only, score it on fold k, then
pool. If every fold picks the same N and the pooled score matches the in-sample one, the
threshold is a stable property of the model rather than a fitted parameter.

Needs only the sweep's per_case.json and the run's splits_final.json — no images, no GPU.
Regions and thresholds are read off the per_case keys, so it works unchanged on any
region set the sweep was run with (composites or raw labels).

    python nnunet_native/crossfit_lesionwise_threshold.py \
        --per_case <sweep_output>/per_case.json \
        --splits <nnUNet_preprocessed>/DatasetXXX/splits_final.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

KEY = re.compile(r"^([^|]+)\|(\d+)\|lw_dice$")


def _detect(cases):
    """Regions and thresholds actually present in the per-case records."""
    regions, thresholds = [], set()
    for k in cases[0]:
        m = KEY.match(k)
        if m:
            if m.group(1) not in regions:
                regions.append(m.group(1))
            thresholds.add(int(m.group(2)))
    return regions, sorted(thresholds)


def _score(subset, regions, n):
    """Mean over regions of (nanmean over cases) — same aggregation as sweep.json."""
    vals = []
    for r in regions:
        xs = [c.get(f"{r}|{n}|lw_dice") for c in subset]
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        vals.append(sum(xs) / len(xs) if xs else float("nan"))
    return sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_case", required=True, help="per_case.json written by the sweep")
    ap.add_argument("--splits", required=True, help="splits_final.json of the swept run")
    args = ap.parse_args()

    cases = json.load(open(args.per_case))
    splits = json.load(open(args.splits))
    regions, thresholds = _detect(cases)

    fold_of = {c: k for k, f in enumerate(splits) for c in f["val"]}
    unmapped = [c["case_id"] for c in cases if c["case_id"] not in fold_of]
    print(f"cases={len(cases)}  regions={regions}  thresholds={thresholds}  "
          f"unmapped={len(unmapped)}")
    if unmapped:
        raise SystemExit(f"ABORT: {len(unmapped)} cases not in any val fold, e.g. {unmapped[0]}"
                         " — wrong splits file?")

    ins = {n: _score(cases, regions, n) for n in thresholds}
    best_in = max(thresholds, key=lambda n: ins[n])
    print("\nin-sample:")
    for n in thresholds:
        print(f"  N={n:>4}  {ins[n]:.4f}" + ("  <- best" if n == best_in else ""))

    print("\ncross-fitted (N chosen without ever seeing the fold it scores):")
    chosen = {}
    for k in range(len(splits)):
        tr = [c for c in cases if fold_of[c["case_id"]] != k]
        te = [c for c in cases if fold_of[c["case_id"]] == k]
        chosen[k] = max(thresholds, key=lambda n: _score(tr, regions, n))
        print(f"  fold {k}: chose N={chosen[k]:>4} on {len(tr)} cases -> "
              f"{_score(te, regions, chosen[k]):.4f} on its {len(te)} held-out "
              f"(raw {_score(te, regions, 0) if 0 in thresholds else float('nan'):.4f})")

    # pooled: each case scored under its own fold's cross-fitted N
    pooled = []
    for c in cases:
        n = chosen[fold_of[c["case_id"]]]
        pooled.append({f"{r}|X|lw_dice": c.get(f"{r}|{n}|lw_dice") for r in regions})
    cf = _score(pooled, regions, "X")

    print(f"\npooled cross-fitted : {cf:.4f}")
    print(f"in-sample best      : {ins[best_in]:.4f}   (N={best_in})")
    print(f"optimism            : {ins[best_in] - cf:+.4f}")
    if 0 in thresholds:
        print(f"raw baseline        : {ins[0]:.4f}")
        print(f"honest gain from PP : {cf - ins[0]:+.4f}")


if __name__ == "__main__":
    main()
