#!/usr/bin/env bash
# Queue for the exp20-vs-exp22 loss comparison on the local 8 GB box.
#
# Waits for the currently-running exp20 (Dice-CE) container to exit, checks it actually
# reached 250 epochs, then launches exp22 (Dice-Focal). Runs must be sequential — only one
# ResEnc-M fits in 8 GB.
#
# Launch detached:
#   nohup ./run_loss_comparison_queue.sh > logs/queue.log 2>&1 &
#
# If exp20 did NOT finish (crash, power cut, OOM) this script stops and leaves the GPU
# free, so you can resume exp20 with replica-train-resume instead of losing its progress.
# See LOCAL_RUN_STATE.md.

set -u
cd "$(dirname "$0")"

REPO="$(pwd)"
export REPLICA_CACHE=/home/zain/brats_replica_cache
EXP20_DIR="runs/replica_nnunet_replica_resenc_m_11g_250ep_20260814_051319_fold0"
TARGET_EPOCHS=250

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }

log "queue started; waiting for exp20 container to exit"

# Wait while any replica-train container is up.
while docker ps --format '{{.Names}}' | grep -q 'replica-train'; do
    sleep 120
done

log "no replica-train container running"

# nnU-Net's log records one epoch_end_timestamps line per finished epoch.
last_epoch=$(grep -c 'epoch_end_timestamps' "$EXP20_DIR/training.log" 2>/dev/null || echo 0)
log "exp20 finished epochs: $last_epoch / $TARGET_EPOCHS"

if [ "$last_epoch" -lt "$TARGET_EPOCHS" ]; then
    log "exp20 did NOT reach $TARGET_EPOCHS epochs — NOT starting exp22."
    log "Resume exp20 with:"
    log "  REPLICA_CACHE=$REPLICA_CACHE REPLICA_CONFIG=exp20_replica_resenc_m_11g_250ep_5fold.yaml \\"
    log "  REPLICA_FOLD=0 REPLICA_ARGS=\"--num_workers 14\" docker-compose run --rm -T replica-train-resume"
    exit 1
fi

log "exp20 complete. Launching exp22 (Dice-Focal)."

REPLICA_CACHE=/home/zain/brats_replica_cache \
REPLICA_CONFIG=exp22_replica_dicefocal_250ep_5fold.yaml \
REPLICA_FOLD=0 \
REPLICA_ARGS="--num_workers 14" \
  docker-compose run --rm -T replica-train \
  > logs/exp22_250ep_fold0_docker.log 2>&1

rc=$?
log "exp22 exited with code $rc"
log "Both arms done — run the evaluation in LOCAL_RUN_STATE.md section 7."
