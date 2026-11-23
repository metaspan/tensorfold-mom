#!/usr/bin/env bash
# MoM MVP serving invocation (P2, 23 Nov 2026).
#
# Canonical config: BOTH lanes in request-starts prompt-cache mode
# (TF_QWEN_REQUEST_STARTS=1) + exact re-prefill on lane switch. This is the
# adopted MVP from docs-mom/docs/12-shared-kv-plan.md ("MVP adoption"
# addendum): lane-local caching only; cross-checkpoint shared cache is closed
# (S1 run 3 NO-GO). Measured: warm same-lane reuse 100%/100%, prefill saved
# 99.0%, TTFT p50 9.8 ms (2B) / 27.4 ms (35B) -- spike-2 rerun,
# docs-mom/gates/20261123-spike2-rerun.md. Escalation cost bound: heavy
# delta-prefill p90 3.3 s at multi-k blocks -- docs-mom/gates/20261124-b5-long.md.
#
# The boot assertion below refuses to proceed if the engine did not register
# the request-starts override (BL-4 will move this check into mom-serve).
#
# On-device only (GB10). --parallel 2 currently fails this device/build at any
# context >= 4k (cannot reserve 2 streams' caches), so parallel is 1 with
# both lanes resident.
set -euo pipefail

BIN="${TFM_BIN:-/home/derek/tensorfold-mom/zig-out/native/bin/tensorfold-native}"
SMALL="${TFM_SMALL:-/home/derek/.hermes/models/tensorfold-mom/qwen35-2b}"
BIG="${TFM_BIG:-/home/derek/.hermes/models/tensorfold-mom/qwen36-35b-a3b-nvfp4}"
SMALL_NAME="${TFM_SMALL_NAME:-qwen3.5-2b}"
BIG_NAME="${TFM_BIG_NAME:-qwen3.6-35b}"
PORT="${TFM_PORT:-8218}"
CONTEXT="${TFM_CONTEXT:-4096}"
LOG="${TFM_LOG:-/tmp/mom-mvp.log}"

export TF_QWEN_REQUEST_STARTS=1

"$BIN" mom-serve \
  --host 127.0.0.1 --port "$PORT" \
  --parallel 1 --no-sticky --context "$CONTEXT" \
  --model "$SMALL" --name "$SMALL_NAME" --lane trivial --pin \
  --model "$BIG"   --name "$BIG_NAME"   --lane default \
  > "$LOG" 2>&1 &
SRV=$!

# Boot assertion: both lanes loaded, routing on 2 lanes, cache mode registered.
deadline=$((SECONDS + 900))
while [ $SECONDS -lt $deadline ]; do
  if ! kill -0 $SRV 2>/dev/null; then
    echo "mom-serve died; log tail:"; tail -20 "$LOG" >&2; exit 2
  fi
  if grep -q "routing over 2 lanes" "$LOG" 2>/dev/null && \
     grep -q "registered overrides: TF_QWEN_REQUEST_STARTS=1" "$LOG"; then
    echo "MVP boot OK: both lanes up, request-starts cache mode registered."
    wait $SRV
    exit 0
  fi
  sleep 2
done
echo "boot assertion timed out waiting for cache-mode registration line; log tail:" >&2
tail -20 "$LOG" >&2
kill $SRV 2>/dev/null || true
exit 3
