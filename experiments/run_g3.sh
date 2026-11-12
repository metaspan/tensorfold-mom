#!/usr/bin/env bash
# G3 device window: build-freshness-checked, then the adoption gate probe.
set -o pipefail
BIN=/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native
STAGE=$(stat -c %Y "$BIN")
NOW=$(date +%s)
AGE=$(( NOW - STAGE ))
if [ "$AGE" -gt 3600 ]; then echo "REFUSE: binary is $AGE s old -- build gate did not land"; exit 2; fi
# no stale MoM servers on the probe port band
if ss -ltn 2>/dev/null | grep -q ':8218 '; then echo "REFUSE: something holds :8218"; ss -ltnp 2>/dev/null | grep ':8218 ' || true; exit 2; fi
cd /home/derek/tensorfold-mom/experiments
/home/derek/.venv/bin/python m3_adoption_probe.py 8 2>&1 | tee /tmp/g3-run.log
echo "G3_RC=$?"
