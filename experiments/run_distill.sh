#!/usr/bin/env bash
# Distillation window (docs/13 part 3): heavy rollouts -> student tail SFT.
# Collect boots its own single-lane mom-serve on :8217 (refuses if held);
# train runs after and must not start if collect failed (Derek's rule:
# failures block the next step).
set -o pipefail
cd /home/derek/tensorfold-mom/experiments
PY=/home/derek/.venv/bin/python
OUT_DIR=results/distill
mkdir -p "$OUT_DIR"
ROWS="$OUT_DIR/rollouts.jsonl"
if ss -ltn 2>/dev/null | grep -q ':8217 '; then echo "PORT-BUSY-8217"; exit 3; fi
"$PY" distill_collect.py --out "$ROWS" --passes 8
COLLECT_RC=$?
echo "COLLECT_RC=$COLLECT_RC rows=$(wc -l < "$ROWS" 2>/dev/null || echo 0)"
[ "$COLLECT_RC" -ne 0 ] && exit "$COLLECT_RC"
"$PY" distill_train.py --data "$ROWS" --out-dir /home/derek/models/tensorfold-mom/qwen36-nested-16l-t1
echo "TRAIN_RC=$?"
echo DISTILL-DONE
