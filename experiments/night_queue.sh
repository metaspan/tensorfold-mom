#!/usr/bin/env bash
# Night queue 2026-11-14: t4 KD training -> gate -> D2 -> MTP trained-head preview.
# Sequential GPU steps (unified-RAM discipline: never two >20GB jobs at once).
set -u
EXP=/home/derek/tensorfold-mom/src/tensorfold/experiments
PY=/home/derek/.venv/bin/python
MODELS=/home/derek/models/tensorfold-mom
BIN=/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native
ST=$EXP/results/distill
log(){ echo "[$(date +%H:%M:%S)] $*" >> /tmp/night-queue.status; }

# S0: wait for D1 to finish (it holds the GPU now)
for i in $(seq 1 180); do grep -q D1-DONE /tmp/d1.out 2>/dev/null && break; sleep 30; done
log "S0 D1 done (rc=$(grep -o 'D1_RC=[0-9]*' /tmp/d1.out || echo '?'))"

# S1: t4 KD training (pre-registered: beta 0.5, T 1.0, k=8 file, 3 epochs, seed 7)
cd "$EXP"
$PY distill_train.py --data results/distill/rollouts-10k-clean.jsonl \
    --out-dir $MODELS/qwen36-nested-16l-t4 --epochs 3 --seed 7 \
    --kd results/distill/kd-10k.pt --beta 0.5 --kd-temp 1.0 \
    > /tmp/train-t4.out 2>&1
rc=$?; log "S1 t4 train rc=$rc (tail: $(tail -2 /tmp/train-t4.out | tr '\n' ' ' | cut -c1-160))"
[ $rc -ne 0 ] && log "S1-FAIL t4 training — continuing queue"

# S2: t4 gate (HF stack; bf16 MoE refused by native)
if [ -d $MODELS/qwen36-nested-16l-t4 ]; then
  $PY distill_gate.py --engine hf --model $MODELS/qwen36-nested-16l-t4 \
      > /tmp/gate-t4.out 2>&1
  log "S2 gate-t4 rc=$? :: $(grep -a 'GATE' /tmp/gate-t4.out | tail -1)"
  grep -qa 'PASS' /tmp/gate-t4.out && T4_PASS=1 || T4_PASS=0
else
  T4_PASS=0; log "S2 skipped (no t4 dir)"
fi

# S3: D2 early-exit baseline at layer 10
$PY d2_early_exit.py --teacher $MODELS/qwen36-35b-a3b-fp8-deq \
    --out $ST/d2-early-exit.json > /tmp/d2.out 2>&1
log "S3 D2 rc=$? :: $(grep -a 'EARLY-EXIT' /tmp/d2.out | tail -1)"

# S4: MTP head trained on t3 (distribution preview, pre-registered)
$PY mtp_train.py --student $MODELS/qwen36-nested-16l-t3 \
    --data results/distill/rollouts-10k-clean.jsonl \
    --out-dir $MODELS/mtphead-t3 \
    > /tmp/mtphead-t3.out 2>&1
log "S4 mtphead-t3 train rc=$? (tail: $(tail -2 /tmp/mtphead-t3.out | tr '\n' ' ' | cut -c1-160))"

# S5+S6: merge head into t3 and serve, probe (thinking off)
if [ -d $MODELS/mtphead-t3 ]; then
  $PY mtp_merge.py --parent $MODELS/mtphead-t3 --nested $MODELS/qwen36-nested-16l-t3 \
      --out $MODELS/qwen36-nested-16l-t3-mtp > /tmp/mtp-merge-t3.out 2>&1
  log "S5 merge rc=$?"
  if [ -d $MODELS/qwen36-nested-16l-t3-mtp ]; then
    "$BIN" serve "$MODELS/qwen36-nested-16l-t3-mtp" --name mtp-t3 --port 8219 \
        --no-thinking > /tmp/mtp-serve-t3.log 2>&1 &
    SPID=$!
    for i in $(seq 1 60); do grep -qa "serving mtp-t3" /tmp/mtp-serve-t3.log 2>/dev/null && break; sleep 2; done
    if grep -qa "serving mtp-t3" /tmp/mtp-serve-t3.log 2>/dev/null; then
      $PY mtp_probe2.py 8219 mtp-t3 12 > /tmp/mtp-probe-t3.out 2>&1
      log "S6 probe-t3 :: $(grep -a mean_tok_s /tmp/mtp-probe-t3.out | tail -1)"
    else
      log "S6-FAIL serve did not come up: $(tail -3 /tmp/mtp-serve-t3.log | tr '\n' ' ' | cut -c1-200)"
    fi
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    grep -a "done " /tmp/mtp-serve-t3.log > /tmp/mtp-done-t3.txt 2>/dev/null || true
  fi
fi

# S7: if t4 passed the gate, train + measure the head on the GATED student
if [ "${T4_PASS:-0}" = "1" ]; then
  $PY mtp_train.py --student $MODELS/qwen36-nested-16l-t4 \
      --data results/distill/rollouts-10k-clean.jsonl \
      --out-dir $MODELS/mtphead-t4 > /tmp/mtphead-t4.out 2>&1
  log "S7 mtphead-t4 train rc=$?"
  if [ -d $MODELS/mtphead-t4 ]; then
    $PY mtp_merge.py --parent $MODELS/mtphead-t4 --nested $MODELS/qwen36-nested-16l-t4 \
        --out $MODELS/qwen36-nested-16l-t4-mtp > /tmp/mtp-merge-t4.out 2>&1
    if [ -d $MODELS/qwen36-nested-16l-t4-mtp ]; then
      "$BIN" serve "$MODELS/qwen36-nested-16l-t4-mtp" --name mtp-t4 --port 8219 \
          --no-thinking > /tmp/mtp-serve-t4.log 2>&1 &
      SPID=$!
      for i in $(seq 1 60); do grep -qa "serving mtp-t4" /tmp/mtp-serve-t4.log 2>/dev/null && break; sleep 2; done
      if grep -qa "serving mtp-t4" /tmp/mtp-serve-t4.log 2>/dev/null; then
        $PY mtp_probe2.py 8219 mtp-t4 12 > /tmp/mtp-probe-t4.out 2>&1
        log "S7 probe-t4 :: $(grep -a mean_tok_s /tmp/mtp-probe-t4.out | tail -1)"
      else
        log "S7-FAIL serve t4 down: $(tail -3 /tmp/mtp-serve-t4.log | tr '\n' ' ' | cut -c1-200)"
      fi
      kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
      grep -a "done " /tmp/mtp-serve-t4.log > /tmp/mtp-done-t4.txt 2>/dev/null || true
    fi
  fi
fi

log "NIGHT-QUEUE-DONE"
echo NIGHT-QUEUE-DONE
