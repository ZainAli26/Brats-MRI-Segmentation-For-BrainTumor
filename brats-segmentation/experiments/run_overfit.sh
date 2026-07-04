#!/usr/bin/env bash
# ==============================================================================
# Run the 50-MRI overfit sanity check for one experiment (or all).
#
# Each overfit config trains + validates + tests on the SAME 50 cases with
# augmentation and early-stopping disabled. A healthy model/pipeline should push
# region Dice on this subset to ~1.0; if it plateaus low, something upstream
# (labels, loss, class count, model wiring) is broken.
#
# Usage:
#   bash experiments/run_overfit.sh 1          # overfit exp01
#   bash experiments/run_overfit.sh 18         # overfit exp18
#   bash experiments/run_overfit.sh all        # overfit every experiment, in order
#   bash experiments/run_overfit.sh 1 --epochs 50   # extra args forwarded to train.py
#
# Configs live in experiments/overfit/ (regenerate with generate_overfit_configs.py).
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

run_one() {
    local num=$1; shift || true
    local pattern="experiments/overfit/overfit_exp$(printf '%02d' "$num")_*.yaml"
    local config
    config=$(ls $pattern 2>/dev/null | head -1)
    if [ -z "$config" ]; then
        echo "No overfit config found for experiment $num ($pattern)"
        return 1
    fi
    mkdir -p runs/overfit
    echo ""
    echo "================================================================"
    echo "  OVERFIT exp$num: $config"
    echo "  Started: $(date)"
    echo "================================================================"
    python train.py --config "$config" "$@" 2>&1 \
        | tee "runs/overfit/overfit_exp$(printf '%02d' "$num")_$(date +%Y%m%d_%H%M%S).log"
    echo "  Finished: $(date)"
}

main() {
    if [ $# -lt 1 ]; then
        echo "Usage: bash experiments/run_overfit.sh <exp-number|all> [extra train.py args]"
        exit 1
    fi

    local target=$1; shift || true

    if [ "$target" = "all" ]; then
        for config in experiments/overfit/overfit_exp*.yaml; do
            num=$(basename "$config" | sed -E 's/overfit_exp0?([0-9]+)_.*/\1/')
            run_one "$num" "$@"
        done
    else
        run_one "$target" "$@"
    fi
}

main "$@"
