#!/usr/bin/env python3
"""Generate per-experiment overfitting configs.

For every experiments/expNN_*.yaml this writes experiments/overfit/overfit_<name>.yaml,
a sanity-check config that trains + validates + tests on the SAME 50-MRI subset with
augmentation and early-stopping disabled (handled by train.py's overfit hook). A
correct model/pipeline should drive region Dice on this subset to ~1.0; if it can't,
something upstream (labels, loss, model wiring) is broken.

Run from the brats-segmentation/ directory:
    python experiments/generate_overfit_configs.py

These outputs are deterministic — regenerate any time the base experiment configs change.
"""
import glob
import os
import re
import yaml

# Keys train.py / the Trainer read from the training block. If a source config has a
# stub training block (e.g. exp19, written for native nnU-Net), the overfit config
# would crash on a KeyError — so we validate every generated config up front.
REQUIRED_TRAINING_KEYS = [
    "epochs", "batch_size", "num_workers", "learning_rate", "weight_decay",
    "optimizer", "scheduler", "amp", "val_interval", "early_stopping_patience",
    "loss", "sw_batch_size", "sw_overlap",
]
REQUIRED_DATA_KEYS = [
    "train_dir", "modalities", "num_classes", "label_map",
    "split_ratios", "split_seed",
]

OVERFIT_EPOCHS = 150
OVERFIT_VAL_INTERVAL = 10
OVERFIT_OPTIMIZER = "adamw"   # SGD@1e-2 background-collapses at batch 1 in this loop; AdamW learns
OVERFIT_WARMUP = 0

# The overfit is a WIRING sanity check ("can the model/loss/labels fit at all"), NOT a
# recipe test — it overrides the optimizer/LR/schedule and strips inference tricks. The
# REAL configs keep their own recipe (e.g. exp19 stays nnU-Net SGD); only overfit deviates.
#
# Two tiers:
#   * DEFAULT (exp01-17): 50 cases, AdamW, source schedule, TTA/postproc off.
#   * TUNED (exp18-25, the active experiment set): 20 cases + CONSTANT LR (a decaying LR
#     stalls memorization before it fits) + a higher LR for fast overfit.
# Per-architecture LR (AdamW): SegResNet diverges to NaN above ~1e-4 (verified on exp21),
# so it is pinned to 1e-4; ResEnc tolerates more.
DEFAULT_CASES, TUNED_CASES = 50, 20
SEG_LR = 0.0001                       # SegResNet: NaN above this
RESENC_LR_DEFAULT, RESENC_LR_TUNED = 0.001, 0.001   # 0.002 diverged to NaN on the ~102M ResEnc (AMP)

HEADER = (
    "# ============================================================\n"
    "# AUTO-GENERATED OVERFIT CONFIG — do not edit by hand.\n"
    "# Source: {src}\n"
    "# Regenerate: python experiments/generate_overfit_configs.py\n"
    "#\n"
    "# Sanity check: memorize {n} MRIs (train = val = test, augmentation OFF,\n"
    "# early-stopping OFF). Watch TRAIN LOSS -> ~0.1 (val Dice is capped by the\n"
    "# train-crop vs sliding-window-val mismatch).\n"
    "# Overrides vs source: {ov}.\n"
    "# ============================================================\n"
)


def _exp_num(name):
    m = re.search(r'exp(\d+)', name)
    return int(m.group(1)) if m else -1


def transform(text, src_name):
    src_cfg = yaml.safe_load(text)
    model_name = str((src_cfg.get("model") or {}).get("name", "")).lower()
    is_seg = model_name == "segresnet"
    is_nnunet = model_name.startswith("nnunet_v2")
    tuned = 18 <= _exp_num(src_name) <= 25          # the active experiment set

    cases = TUNED_CASES if tuned else DEFAULT_CASES
    lr = SEG_LR if is_seg else (RESENC_LR_TUNED if tuned else RESENC_LR_DEFAULT)
    has_gradckpt = "grad_checkpointing" in text     # don't inject a duplicate key

    lines = text.splitlines(keepends=True)
    out = []
    for line in lines:
        new = line
        if re.match(r'\s*epochs:\s*\d+', new):
            new = re.sub(r'epochs:\s*\d+', f'epochs: {OVERFIT_EPOCHS}', new)
        elif re.match(r'\s*val_interval:\s*\d+', new):
            new = re.sub(r'val_interval:\s*\d+', f'val_interval: {OVERFIT_VAL_INTERVAL}', new)
        elif re.match(r'\s*output_dir:\s*["\']\./runs["\']', new):
            new = re.sub(r'(["\'])\./runs\1', r'\1./runs/overfit\1', new)
        # Override optimizer/LR/warmup. Matches only ACTIVE lines (commented alternatives
        # start with '#' and fail the leading-`key:` match).
        elif re.match(r'\s*optimizer:\s*["\']?\w+', new):
            new = re.sub(r'optimizer:\s*["\']?\w+["\']?', f'optimizer: "{OVERFIT_OPTIMIZER}"', new)
        elif re.match(r'\s*learning_rate:\s*[\d.eE+-]+', new):
            new = re.sub(r'learning_rate:\s*[\d.eE+-]+', f'learning_rate: {lr}', new)
        elif re.match(r'\s*warmup_epochs:\s*\d+', new):
            new = re.sub(r'warmup_epochs:\s*\d+', f'warmup_epochs: {OVERFIT_WARMUP}', new)
        # Drop nnU-Net's fixed-iteration epochs: the overfit is a full-pass memorization
        # over a tiny set, so revert to one pass/epoch (Trainer falls back when absent).
        elif re.match(r'\s*iters_per_epoch:\s*\d+', new):
            continue
        # TUNED tier: constant LR so memorization doesn't stall as the schedule decays.
        elif tuned and re.match(r'\s*scheduler:\s*["\']?\w+', new):
            new = re.sub(r'scheduler:\s*["\']?\w+["\']?', 'scheduler: "constant"', new)
        # Strip inference tricks -> raw val metric (postprocess otherwise zeroes small ET).
        elif re.match(r'\s*tta:\s*true', new):
            new = re.sub(r'tta:\s*true', 'tta: false', new)
        elif re.match(r'\s*postprocess:\s*true', new):
            new = re.sub(r'postprocess:\s*true', 'postprocess: false', new)
        out.append(new)
        # Enable overfit mode right after class_names (present in every migrated config).
        if re.match(r'\s*class_names:', line):
            out.append(f"  overfit: {{enabled: true, num_cases: {cases}}}\n")
        # Inject grad_checkpointing for nnU-Net v2 (after deep_supervision) so AdamW + the
        # big patch fits 8 GB. Skip if the source already declares it (no duplicate key).
        if is_nnunet and not has_gradckpt and re.match(r'\s*deep_supervision:\s*true', line):
            indent = line[:len(line) - len(line.lstrip())]
            out.append(f"{indent}grad_checkpointing: true   # overfit: fit AdamW + big patch on 8 GB\n")
    body = "".join(out)
    if "overfit:" not in body:
        raise RuntimeError(f"{src_name}: no class_names anchor found — cannot insert overfit block")
    ov = (f"optimizer=AdamW, lr={lr}, " + ("scheduler=constant, " if tuned else "")
          + f"warmup=0, cases={cases}, TTA=off, postprocess=off")
    return HEADER.format(src=src_name, n=cases, ov=ov) + body


def main():
    os.makedirs("experiments/overfit", exist_ok=True)
    srcs = sorted(p for p in glob.glob("experiments/*.yaml"))
    bad = {}
    for src in srcs:
        name = os.path.basename(src)
        with open(src) as f:
            text = f.read()
        dst = f"experiments/overfit/overfit_{name}"
        out = transform(text, name)
        with open(dst, "w") as f:
            f.write(out)
        # Validate the result is actually runnable through train.py.
        cfg = yaml.safe_load(out)
        missing = [f"training.{k}" for k in REQUIRED_TRAINING_KEYS if k not in cfg.get("training", {})]
        missing += [f"data.{k}" for k in REQUIRED_DATA_KEYS if k not in cfg.get("data", {})]
        # Preprocessing must be concrete, not a "not enforced" stub.
        prep = cfg.get("preprocessing", {})
        if not isinstance(prep.get("spatial_size"), list):
            missing.append("preprocessing.spatial_size (must be a list, not a stub string)")
        if not isinstance(prep.get("augmentation"), dict):
            missing.append("preprocessing.augmentation (must be a dict, not a stub string)")
        # Model must be one the factory can build.
        mname = str(cfg.get("model", {}).get("name", "")).lower()
        if not (mname.startswith("nnunet_v2") or mname in ("dynunet", "swin_unetr", "segresnet")):
            missing.append(f"model.name '{mname}' not recognized by the factory")
        flag = f"  [!] NOT runnable — missing {missing}" if missing else ""
        print(f"wrote {dst}{flag}")
        if missing:
            bad[name] = missing

    print(f"\n{len(srcs)} overfit configs generated in experiments/overfit/")
    if bad:
        print("\nWARNING: these base configs are incomplete for train.py — fix them and re-run:")
        for name, missing in bad.items():
            print(f"  {name}: missing {missing}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
