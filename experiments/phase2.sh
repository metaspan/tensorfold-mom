#!/usr/bin/env bash
# Phase 2 (overnight 2026-11-15/16): 5e last-6 byte-copy gate + vectorized MTP head.
# GPU-serial. Hourly HEARTBEAT lines to stdout (notify pattern).
set -u
D=/home/derek/tensorfold-mom/src/tensorfold/experiments
MODELS=/home/derek/models/tensorfold-mom
ST=$D/results
PY=/home/derek/.venv/bin/python
BIN=/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native
cd "$D" || exit 1
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# hourly heartbeat while stages run
( while true; do sleep 3600; echo "HEARTBEAT $(date -u +%H:%M:%S) :: $(tail -1 /tmp/phase2.status 2>/dev/null)"; done ) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT
st(){ echo "[$(date +%H:%M:%S)] $*" >> /tmp/phase2.status; }

# ---- P0: last6 byte-for-byte verification (5e pre-registration) ----
timeout 1800 $PY last6_verify.py > /tmp/last6-verify.out 2>&1; P0RC=$?
P0OK=0
grep -qa "mismatched 0 missing-parent 0" /tmp/last6-verify.out && P0OK=1
st "P0 verify rc=$P0RC ok=$P0OK :: $(grep -a 'tail bytes' /tmp/last6-verify.out | tail -1 | cut -c1-120)"

# ---- P1: native coherence gate on last6 (bar 10/12 pre-registered) ----
if [ "$P0OK" = 1 ]; then
  timeout 1500 $PY distill_gate.py --engine native --model $MODELS/qwen36-nested-16l-last6 \
      > /tmp/last6-gate.out 2>&1; RC=$?
  st "P1 last6 gate rc=$RC :: $(grep -a 'GATE ' /tmp/last6-gate.out | tail -1 | cut -c1-120)"
else
  st "P1 SKIPPED (P0 not clean)"
fi

# ---- P2: MTP head on t3 with vectorized MoE (fresh dir) ----
rm -rf $MODELS/mtphead-t3
timeout 21600 $PY mtp_train.py --student $MODELS/qwen36-nested-16l-t3 \
    --data results/distill/rollouts-10k-clean.jsonl \
    --out-dir $MODELS/mtphead-t3 \
    > /tmp/mtphead-t3.out 2>&1
st "P2 mtphead-t3 train rc=$? (tail: $(tail -2 /tmp/mtphead-t3.out | tr '\n' ' ' | cut -c1-160))"

# ---- P3: merge head into t3, serve native, probe thinking-off ----
if [ -d $MODELS/mtphead-t3 ]; then
  timeout 600 $PY mtp_merge.py --parent $MODELS/mtphead-t3 --nested $MODELS/qwen36-nested-16l-t3 \
      --out $MODELS/qwen36-nested-16l-t3-mtp > /tmp/mtp-merge-t3.out 2>&1
  st "P3a merge rc=$? :: $(tail -1 /tmp/mtp-merge-t3.out | cut -c1-140)"
  pkill -f "qwen36-nested-16l-t3-mt[p]" 2>/dev/null; sleep 2
  "$BIN" serve "$MODELS/qwen36-nested-16l-t3-mtp" --name mtp-t3 --port 8219 \
      --no-thinking > /tmp/mtp-serve-t3.log 2>&1 &
  SRV=$!
  for i in $(seq 1 90); do grep -qa "serving mtp-t3" /tmp/mtp-serve-t3.log 2>/dev/null && break; sleep 2; done
  if grep -qa "serving mtp-t3" /tmp/mtp-serve-t3.log 2>/dev/null; then
    timeout 900 $PY mtp_probe2.py 8219 mtp-t3 12 > /tmp/mtp-probe-t3.out 2>&1
    st "P3 probe rc=$? :: $(grep -a 'tok/s' /tmp/mtp-probe-t3.out | tail -1 | cut -c1-160)"
    grep -a "done " /tmp/mtp-serve-t3.log > /tmp/mtp-done-t3.txt 2>/dev/null || true
  else
    st "P3-FAIL serve did not come up: $(tail -3 /tmp/mtp-serve-t3.log | tr '\n' ' ' | cut -c1-200)"
  fi
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
else
  st "P3 SKIPPED (no mtphead-t3)"
fi

st "PHASE2-COMPLETE"
echo "PHASE2-DONE"
