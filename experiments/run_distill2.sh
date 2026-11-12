#!/usr/bin/env bash
# Distillation v2 device window: collect (96-seed corpus, thinking rows) ->
# tail SFT (eval split) -> PRE-REGISTERED coherence gate. Launch only when the
# probe band is free (M4 window finished) and the native binary is fresh.
set -o pipefail
cd /home/derek/tensorfold-mom/experiments
BIN=/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native
STAGE=$(stat -c %Y "$BIN"); NOW=$(date +%s); AGE=$(( NOW - STAGE ))
if [ "$AGE" -gt 3600 ]; then echo "REFUSE: binary is $AGE s old"; exit 2; fi
if ss -ltn 2>/dev/null | grep -qE ':821[789] '; then echo "REFUSE: probe/collect/gate port band busy"; exit 2; fi
PY=/home/derek/.venv/bin/python
OUT=results/distill/rollouts_v2.jsonl
mkdir -p results/distill

$PY distill_collect.py --out "$OUT" --passes 8 --thinking-rate 0.25
RC1=$?; echo "COLLECT_RC=$RC1"; [ $RC1 -ne 0 ] && exit 1
ROWS=$(wc -l < "$OUT")
echo "corpus rows: $ROWS"
if [ "$ROWS" -lt 400 ]; then echo "REFUSE: corpus too small ($ROWS) to train t2"; exit 2; fi

$PY distill_train.py --data "$OUT" \
    --out-dir /home/derek/models/tensorfold-mom/qwen36-nested-16l-t2 \
    --epochs 3 --eval-fraction 0.1
RC2=$?; echo "TRAIN_RC=$RC2"; [ $RC2 -ne 0 ] && exit 1

$PY distill_gate.py --model /home/derek/models/tensorfold-mom/qwen36-nested-16l-t2
RC3=$?; echo "GATE_RC=$RC3"
echo "DISTILL2_DONE RC collect=$RC1 train=$RC2 gate=$RC3"
