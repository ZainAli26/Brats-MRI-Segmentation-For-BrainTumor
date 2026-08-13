#!/usr/bin/env python3
"""Is the gradient-checkpointed path equivalent to the stored-activation path?

`equiv_checks.py` proved that `src/nnunet_replica/` computes what native nnU-Net 2.4.2
computes — but every one of those checks ran with `grad_checkpointing=False`
(`equiv_checks.py:130`). Native nnU-Net has no gradient checkpointing at all: grepping
`nnUNetTrainer.py` finds zero uses of `torch.utils.checkpoint`. The wrapper in
`network.py:57` is ours, and it is what lets the 11 GB plan fit an 8 GB card.

So the code path used by every run on an 8 GB card was never covered by the parity proof.
It *should* be exact — `use_reentrant=False` preserves RNG state, this network has no
dropout, and InstanceNorm is deterministic — but recomputation reorders floating-point work,
and "should be" is not "was measured". These checks close that gap.

What is compared: one model, one set of weights, one batch, run both ways. Not two training
runs — those differ by RNG stream and data order, so any gap between them is confounded.

Run from brats-segmentation/ (no native nnU-Net or PYTHONPATH needed — this is
replica-vs-replica):

    python3 tools/replica_parity/ckpt_checks.py
    python3 tools/replica_parity/ckpt_checks.py --batch 2      # the plan's real batch
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

PLANS = REPO / "plans/nnUNetResEncUNetMPlans_8G_cmp.json"
NUM_CLASSES = 5
NUM_CHANNELS = 4
SEED = 1234

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       {detail}" if detail else ""))


def build_pair(plan, batch, device, grad_ckpt_second=True):
    """Two nets with byte-identical weights: one storing activations, one recomputing."""
    from src.nnunet_replica.network import build_network

    torch.manual_seed(SEED)
    plain = build_network(plan, NUM_CHANNELS, NUM_CLASSES,
                          deep_supervision=True, grad_checkpointing=False).to(device)
    torch.manual_seed(SEED)
    ckpt = build_network(plan, NUM_CHANNELS, NUM_CLASSES,
                        deep_supervision=True, grad_checkpointing=grad_ckpt_second).to(device)
    # Do not rely on the seed alone — copy, so init can never be the explanation.
    ckpt.load_state_dict(plain.state_dict())
    return plain, ckpt


def fixed_batch(plan, batch, device):
    g = torch.Generator(device="cpu").manual_seed(7)
    x = torch.randn(batch, NUM_CHANNELS, *plan.patch_size, generator=g)
    return x.to(device)


def synthetic_objective(outputs, device):
    """A scalar that depends on every deep-supervision head.

    Deliberately not the real Dice+CE loss: this isolates the backward graph from the loss
    implementation, so a failure can only mean the recomputation changed something.

    `mean`, not `sum`, is load-bearing. Summing over five full-resolution heads yields
    gradients that overflow fp16, so GradScaler skips every optimiser step and the
    weights-after-N-steps check passes vacuously — it compares two nets that never moved.
    """
    g = torch.Generator(device="cpu").manual_seed(11)
    total = 0.0
    for o in outputs:
        w = torch.randn(o.shape, generator=g).to(device)
        total = total + (o * w).mean()
    return total


def max_grad_diff(a, b):
    """Absolute and *relative* worst-case gradient difference.

    Relative matters: an absolute 1e-1 is structural on gradients of size 1e-1 and pure
    rounding on gradients of size 1e4.
    """
    ga = {n: p.grad for n, p in a.named_parameters() if p.grad is not None}
    gb = {n: p.grad for n, p in b.named_parameters() if p.grad is not None}
    if set(ga) != set(gb):
        return float("inf"), float("inf"), 0
    worst = scale = 0.0
    for n in ga:
        worst = max(worst, float((ga[n].float() - gb[n].float()).abs().max()))
        scale = max(scale, float(ga[n].float().abs().max()))
    rel = worst / scale if scale > 0 else 0.0
    return worst, rel, sum(g.numel() for g in ga.values())


def max_weight_diff_vs(before: dict, net):
    """How far a net moved from a saved state_dict — guards against vacuous passes."""
    sd = net.state_dict()
    worst = 0.0
    for k in before:
        worst = max(worst, float((before[k].float() - sd[k].float()).abs().max()))
    return worst, len(before)


def max_weight_diff(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    if list(sa) != list(sb):
        return float("inf"), 0
    worst = 0.0
    for k in sa:
        worst = max(worst, float((sa[k].float() - sb[k].float()).abs().max()))
    return worst, sum(v.numel() for v in sa.values())


# --------------------------------------------------------------------------- checks
def check_forward_eval(plan, batch, device):
    plain, ckpt = build_pair(plan, batch, device)
    x = fixed_batch(plan, batch, device)
    plain.eval(); ckpt.eval()
    with torch.no_grad():
        a, b = plain(x), ckpt(x)
    d = max(float((p - q).abs().max()) for p, q in zip(a, b))
    report("forward, eval mode, fp32 — bitwise", d == 0.0,
           f"heads {len(a)}; max|diff| = {d:.3e}")


def check_grads_fp32(plan, batch, device):
    """The check that matters: checkpointing only changes the BACKWARD pass.

    cudnn.deterministic is forced here so that kernel-selection nondeterminism cannot be
    mistaken for a checkpointing difference. This isolates the question being asked.
    """
    prev_bench, prev_det = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        plain, ckpt = build_pair(plan, batch, device)
        x = fixed_batch(plan, batch, device)
        for net in (plain, ckpt):
            net.train()
            net.zero_grad(set_to_none=True)
            synthetic_objective(net(x), device).backward()
        d, rel, n = max_grad_diff(plain, ckpt)
        report("gradients, fp32, cudnn deterministic — bitwise", d == 0.0,
               f"max|grad diff| = {d:.3e} (relative {rel:.3e}) across {n:,} gradient values")
    finally:
        torch.backends.cudnn.benchmark = prev_bench
        torch.backends.cudnn.deterministic = prev_det


def check_grads_amp(plan, batch, device):
    """Same, but under the settings training actually uses: AMP + cudnn.benchmark."""
    torch.backends.cudnn.benchmark = True
    plain, ckpt = build_pair(plan, batch, device)
    x = fixed_batch(plan, batch, device)
    for net in (plain, ckpt):
        net.train()
        net.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=True):
            out = net(x)
            loss = synthetic_objective(out, device)
        loss.float().backward()
    d, rel, n = max_grad_diff(plain, ckpt)
    # AMP reduces in fp16 and cudnn.benchmark may pick a different algorithm for the
    # recomputed forward, so exact zero is not expected. What matters is that the
    # disagreement is at fp16 rounding level relative to gradient magnitude, not structural.
    ok = rel < 1e-2
    report("gradients, AMP + cudnn.benchmark (real training settings)", ok,
           f"max|grad diff| = {d:.3e}, relative = {rel:.3e} across {n:,} values "
           f"(relative tolerance 1e-2; fp16 accumulation order makes exact 0 unlikely)")


def check_weights_after_steps(plan, batch, device, steps=3):
    """Mirrors the shape of the native-parity proof: weights after N real SGD steps."""
    from src.nnunet_replica.network import build_network

    torch.backends.cudnn.benchmark = True
    x = fixed_batch(plan, batch, device)

    # Two nets plus SGD momentum buffers do not fit an 8 GB card together (~1.6 GB of
    # weights and 1.6 GB of buffers before any activation). Build, run and free one at a
    # time, comparing state_dicts held on the CPU.
    torch.manual_seed(SEED)
    init_sd = {k: v.detach().clone() for k, v in
               build_network(plan, NUM_CHANNELS, NUM_CLASSES, deep_supervision=True,
                             grad_checkpointing=False).state_dict().items()}
    before = init_sd

    def run(net):
        opt = torch.optim.SGD(net.parameters(), 1e-2, weight_decay=3e-5,
                              momentum=0.99, nesterov=True)
        scaler = torch.amp.GradScaler(device.type)
        net.train()
        applied = 0
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=True):
                loss = synthetic_objective(net(x), device)
            scaler.scale(loss.float()).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 12.0)
            prev = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # A step is skipped whenever GradScaler sees inf/NaN, which it signals by
            # reducing the scale. Without this, an all-skipped run compares two untouched
            # nets and reports a perfect match.
            applied += int(scaler.get_scale() >= prev)
        return applied

    finals, applied = {}, {}
    for label, use_ckpt in (("plain", False), ("ckpt", True)):
        net = build_network(plan, NUM_CHANNELS, NUM_CLASSES, deep_supervision=True,
                            grad_checkpointing=use_ckpt).to(device)
        net.load_state_dict(init_sd)
        applied[label] = run(net)
        finals[label] = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        del net
        torch.cuda.empty_cache() if device.type == "cuda" else None

    applied_plain, applied_ckpt = applied["plain"], applied["ckpt"]
    moved = max(float((before[k].float() - finals["plain"][k].float()).abs().max())
                for k in before)
    n = sum(v.numel() for v in finals["plain"].values())
    d = max(float((finals["plain"][k].float() - finals["ckpt"][k].float()).abs().max())
            for k in finals["plain"])
    ok = (d < 1e-4 and applied_plain == applied_ckpt == steps and moved > 0)
    report(f"weights after {steps} SGD steps (AMP, grad-clip 12, momentum 0.99)", ok,
           f"max|weight diff| = {d:.3e} across {n:,} values (tolerance 1e-4); "
           f"steps applied {applied_plain}/{steps} vs {applied_ckpt}/{steps}; "
           f"weights moved from init by {moved:.3e} (must be > 0 or this check is vacuous)")


def check_memory(plan, batch, device):
    """Document the number that justifies checkpointing on an 8 GB card."""
    if device.type != "cuda":
        report("peak VRAM, stored vs recomputed", True, "skipped (CPU run)")
        return
    peaks = {}
    for label, use_ckpt in (("stored", False), ("recomputed", True)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        from src.nnunet_replica.network import build_network as _bn
        torch.manual_seed(SEED)
        net = _bn(plan, NUM_CHANNELS, NUM_CLASSES, deep_supervision=True,
                  grad_checkpointing=use_ckpt).to(device)
        x = fixed_batch(plan, batch, device)
        net.train(); net.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=True):
            loss = synthetic_objective(net(x), device)
        loss.float().backward()
        peaks[label] = torch.cuda.max_memory_allocated() / 2 ** 30
        del net, x, loss
    saved = peaks["stored"] - peaks["recomputed"]
    report("peak VRAM, stored vs recomputed activations", saved > 0,
           f"stored {peaks['stored']:.2f} GiB, recomputed {peaks['recomputed']:.2f} GiB, "
           f"saved {saved:.2f} GiB. NOTE: torch counters exclude the CUDA context and cuDNN "
           f"workspaces — nvidia-smi reported ~1.8 GB more than this on a real run.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=1,
                    help="1 (default) is enough for equivalence and fits any card; "
                         "2 is the plan's real batch")
    ap.add_argument("--json", default=None, help="Write results here")
    args = ap.parse_args()

    from src.nnunet_replica.plans import Plans
    plan = Plans.load(PLANS).get_configuration("3d_fullres")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"plan {PLANS.name}: patch {list(plan.patch_size)}, plan batch {plan.batch_size}")
    print(f"running at batch {args.batch} on {device}"
          f"{' — ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}\n")

    check_forward_eval(plan, args.batch, device)
    check_grads_fp32(plan, args.batch, device)
    check_grads_amp(plan, args.batch, device)
    check_weights_after_steps(plan, args.batch, device)
    check_memory(plan, args.batch, device)

    n_pass = sum(r["pass"] for r in RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} checks passed")
    if args.json:
        Path(args.json).write_text(json.dumps(RESULTS, indent=2))
        print(f"wrote {args.json}")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
