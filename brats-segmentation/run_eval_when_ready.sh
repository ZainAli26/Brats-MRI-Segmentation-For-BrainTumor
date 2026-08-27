#!/usr/bin/env bash
# Waits for exp22 to finish, verifies it actually reached 250 epochs, then runs the BraTS
# evaluation on both arms. Completes the pipeline unattended.
#
#   nohup ./run_eval_when_ready.sh > logs/eval_watcher.log 2>&1 &
#
# If exp22 did not finish, this stops WITHOUT evaluating — a partial arm would produce a
# misleading comparison. Resume exp22 first (LOCAL_RUN_STATE.md section 5), then re-run this.

set -u
cd "$(dirname "$0")"

EXP22_DIR="runs/replica_nnunet_replica_resenc_m_11g_dicefocal_250ep_20260817_022925_fold0"
TARGET_EPOCHS=250

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }

log "watcher started; waiting for exp22 to finish"

while docker ps --format '{{.Names}}' | grep -q 'replica-train'; do
    sleep 300
done

log "no replica-train container running"

done_epochs=$(grep -c 'epoch_end_timestamps' "$EXP22_DIR/training.log" 2>/dev/null)
done_epochs=${done_epochs:-0}
log "exp22 finished epochs: $done_epochs / $TARGET_EPOCHS"

if [ "$done_epochs" -lt "$TARGET_EPOCHS" ]; then
    log "exp22 incomplete — NOT evaluating. Resume it first (LOCAL_RUN_STATE.md section 5)."
    exit 1
fi

# Let the GPU settle before the eval script's headroom check.
sleep 60

log "both arms complete — starting BraTS evaluation"
./run_brats_eval.sh
log "eval watcher done (exit $?)"
