#!/usr/bin/env bash
# BraTS evaluation for the exp20-vs-exp22 loss comparison.
#
# Runs the SAME four passes on BOTH arms, so the comparison stays apples-to-apples:
#   1. val  split, raw                       (the headline nnU-Net-style number)
#   2. val  split, + BraTS heuristic cleanup (--postprocess)
#   3. test split, raw                       (seed-42 held-out patients)
#   4. test split, + BraTS heuristic cleanup
#
# Predictions cache to <run_dir>/predictions_<split>/, so passes 2 and 4 reuse pass 1's and
# 3's inference and cost only the post-processing + metrics.
#
# REQUIRES AN IDLE GPU — sliding-window + 8-flip TTA needs the full card. Do not run while
# a training container holds VRAM.
#
#   nohup ./run_brats_eval.sh > logs/brats_eval.log 2>&1 &
#
# TWO POST-PROCESSING FAMILIES — do not conflate them (CLAUDE.md):
#   --postprocess              = BraTS-competition heuristic, hand-picked thresholds.
#                                nnU-Net does NONE of this.
#   --determine_postprocessing = nnU-Net's own data-driven keep-largest-component ops.
# This script runs the BraTS heuristic only. nnU-Net's family is a separate question; if you
# want it, determine on --split val and apply the saved json to --split test, NEVER determine
# on test.

set -u
cd "$(dirname "$0")"
export REPLICA_CACHE=/home/zain/brats_replica_cache

EXP20_DIR="runs/replica_nnunet_replica_resenc_m_11g_250ep_20260814_051319_fold0"
EXP20_CFG="exp20_replica_resenc_m_11g_250ep_5fold.yaml"
EXP22_DIR="runs/replica_nnunet_replica_resenc_m_11g_dicefocal_250ep_20260817_022925_fold0"
EXP22_CFG="exp22_replica_dicefocal_250ep_5fold.yaml"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }

# Refuse to run against a busy GPU rather than dying halfway with an OOM.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 1000 ]; then
    log "ABORT: GPU has ${used} MiB in use — a training run is probably still active."
    log "Wait for it to finish, then re-run this script."
    exit 1
fi

evaluate () {          # $1=run dir  $2=config basename  $3=split  $4=extra flags  $5=label
    local dir="$1" cfg="$2" split="$3" extra="$4" label="$5"
    if [ ! -d "$dir" ]; then log "SKIP $label — run dir missing: $dir"; return 1; fi
    log "=== $label : $cfg / --split $split $extra ==="
    docker-compose run --rm -T replica-train \
        python evaluate_replica.py \
            --config "experiments/$cfg" \
            --run_dirs "$dir" \
            --split "$split" \
            --data_dir /data/Brats2024/training_data1_v2 \
            --extra_data_dir /data/Brats2024/training_data_additional \
            $extra
    log "--- $label done (exit $?) ---"
}

for arm in "exp20:$EXP20_DIR:$EXP20_CFG" "exp22:$EXP22_DIR:$EXP22_CFG"; do
    name="${arm%%:*}"; rest="${arm#*:}"; dir="${rest%%:*}"; cfg="${rest##*:}"
    evaluate "$dir" "$cfg" val  ""              "$name val raw"
    evaluate "$dir" "$cfg" val  "--postprocess" "$name val BraTS-heuristic"
    evaluate "$dir" "$cfg" test ""              "$name test raw"
    evaluate "$dir" "$cfg" test "--postprocess" "$name test BraTS-heuristic"
done

log "All BraTS evaluations complete."
log "Outputs: <run_dir>/replica_metrics_*.csv and <run_dir>/replica_summary_*.json"
