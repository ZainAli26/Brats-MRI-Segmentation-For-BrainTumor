#!/usr/bin/env bash
# ==============================================================================
# Run a single experiment by number (1-13) or phase (p1-p7)
#
# Usage:
#   bash experiments/run_experiment.sh 1        # Run exp01 only
#   bash experiments/run_experiment.sh 5        # Run exp05 only
#   bash experiments/run_experiment.sh p1       # Run all Phase 1 (exp01-04)
#   bash experiments/run_experiment.sh p5       # Run Phase 5 (native nnU-Net)
#   bash experiments/run_experiment.sh eval 1   # Evaluate exp01's run
#   bash experiments/run_experiment.sh all      # Run all experiments sequentially
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

run_exp() {
    local num=$1
    local config="experiments/exp$(printf '%02d' $num)_*.yaml"
    config=$(ls $config 2>/dev/null | head -1)
    if [ -z "$config" ]; then
        echo "No config found for experiment $num"
        return 1
    fi
    echo ""
    echo "================================================================"
    echo "  EXPERIMENT $num: $config"
    echo "  Started: $(date)"
    echo "================================================================"
    python train.py --config "$config" 2>&1 | tee "runs/exp$(printf '%02d' $num)_$(date +%Y%m%d_%H%M%S).log"
    echo "  Finished: $(date)"
}

run_native_nnunet() {
    echo ""
    echo "================================================================"
    echo "  EXPERIMENT 5.1: Native nnU-Net v2 Pipeline"
    echo "  Started: $(date)"
    echo "================================================================"
    bash nnunet_native/run_nnunet.sh --data_dir ../Brats2024/training_data1_v2 --gpu 0
    echo "  Finished: $(date)"
}

eval_exp() {
    local num=$1
    # Find the most recent run dir for this experiment's model
    local config="experiments/exp$(printf '%02d' $num)_*.yaml"
    config=$(ls $config 2>/dev/null | head -1)
    if [ -z "$config" ]; then
        echo "No config for experiment $num"
        return 1
    fi
    local model=$(grep 'name:' "$config" | head -1 | awk '{print $2}' | tr -d '"')
    local run_dir=$(ls -td runs/${model}_* 2>/dev/null | head -1)
    if [ -z "$run_dir" ]; then
        echo "No run found for model $model"
        return 1
    fi
    echo "Evaluating: $run_dir"
    python evaluate.py --run_dir "$run_dir" --visualize_failures
}

case "${1:-help}" in
    # Individual experiments
    [0-9]|[0-9][0-9])
        run_exp "$1"
        ;;

    # Phases
    p1) for i in 1 2 3 4; do run_exp $i; done ;;
    p2) for i in 5 6; do run_exp $i; done ;;
    p3) for i in 7 8; do run_exp $i; done ;;
    p4) for i in 9 10; do run_exp $i; done ;;
    p5) run_native_nnunet ;;
    p6) run_exp 11 ;;
    p7) for i in 12 13; do run_exp $i; done ;;

    # Evaluate
    eval)
        if [ -n "${2:-}" ]; then
            eval_exp "$2"
        else
            echo "Usage: $0 eval <exp_number>"
        fi
        ;;

    # Run everything
    all)
        for i in 1 2 3 4 5 6 7 8 9 10; do run_exp $i; done
        run_native_nnunet
        for i in 11 12 13; do run_exp $i; done
        ;;

    help|*)
        echo "Usage: bash experiments/run_experiment.sh <target>"
        echo ""
        echo "Targets:"
        echo "  1-13       Run a single experiment"
        echo "  p1-p7      Run an entire phase"
        echo "  eval <N>   Evaluate experiment N"
        echo "  all        Run all experiments sequentially"
        echo ""
        echo "Experiments:"
        echo "  Phase 1 — Baseline model comparison"
        echo "    1  segresnet_baseline     (BraTS 2024)"
        echo "    2  dynunet_baseline        (BraTS 2024)"
        echo "    3  nnunet_v2_baseline      (BraTS 2024)"
        echo "    4  swin_unetr_baseline     (BraTS 2024)"
        echo "  Phase 2 — Loss function study"
        echo "    5  dice_ce                 (control)"
        echo "    6  dice_focal              (better for small ET?)"
        echo "  Phase 3 — Input resolution"
        echo "    7  crop 96^3               (faster)"
        echo "    8  crop 160^3              (more context)"
        echo "  Phase 4 — nnU-Net v2 variants"
        echo "    9  nnunet_v2 plain"
        echo "    10 nnunet_v2 residual"
        echo "  Phase 5 — Native nnU-Net v2"
        echo "    p5 native pipeline         (own preprocessing)"
        echo "  Phase 6 — BraTS 2023"
        echo "    11 best model on BraTS 2023"
        echo "  Phase 7 — Cross-dataset"
        echo "    12 train 2024 → test 2023"
        echo "    13 train 2023 → test 2024"
        ;;
esac
