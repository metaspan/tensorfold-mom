#!/usr/bin/env bash
# One-time asset transfer spark-2d3b -> spark4 for the MoM training migration.
# Resumable: rerun after any interruption; rsync picks up where it left off.
set -u
SP4=derek@192.168.1.104
M=/home/derek/models/tensorfold-mom
E=/home/derek/tensorfold-mom/src/tensorfold/experiments
LOG=/home/derek/sp4-sync.log
item(){ # name src dst
  echo "[$(date +%H:%M:%S)] START $1" >> $LOG
  rsync -a --partial --info=stats1 "$2" "$3" >> $LOG 2>&1
  RC=$?
  echo "[$(date +%H:%M:%S)] DONE $1 rc=$RC" >> $LOG
  [ $RC -ne 0 ] && echo "SYNC-FAIL $1 rc=$RC"
  return $RC
}
echo "==== sync session $(date) ====" >> $LOG
# Stage 1: small artifacts (usable immediately by spark4's hermes)
item rollouts "$E/results/distill/rollouts-10k-clean.jsonl" "$SP4:~/tensorfold-mom-data/"
item kd10k    "$E/results/distill/kd-10k.pt"                "$SP4:~/tensorfold-mom-data/"
item night    "$E/results/night"                            "$SP4:~/tensorfold-mom-data/"
item gates    "$E/gate_qa.json"                             "$SP4:~/tensorfold-mom-data/"
item gatejson "$E/results/teacher-gate2.json"               "$SP4:~/tensorfold-mom-data/"
# Stage 2: checkpoints (students + twins)
for d in qwen36-nested-16l-t3 qwen36-nested-16l-t4 qwen36-nested-16l-bf16 qwen36-nested-16l-last6-bf16; do
  item "$d" "$M/$d" "$SP4:~/models/tensorfold-mom/"
done
# Stage 3: the 67 G teacher
item teacher "$M/qwen36-35b-a3b-fp8-deq" "$SP4:~/models/tensorfold-mom/"
# Stage 4: repo bundle (GitHub-auth insurance)
git -C /home/derek/tensorfold-mom/src/tensorfold bundle create /tmp/mom-main.bundle mom/main >> $LOG 2>&1
item bundle /tmp/mom-main.bundle "$SP4:~/tensorfold-mom-data/"
# Spot-verify: one sha256 per shipped dir anchor file, both sides
echo "[$(date +%H:%M:%S)] VERIFY" >> $LOG
for f in qwen36-35b-a3b-fp8-deq/config.json qwen36-nested-16l-t3/config.json; do
  A=$(sha256sum $M/$f | cut -d' ' -f1)
  B=$(ssh $SP4 "sha256sum ~/models/tensorfold-mom/$f | cut -d' ' -f1")
  [ "$A" = "$B" ] && echo "VERIFY-OK $f" >> $LOG || echo "VERIFY-MISMATCH $f" >> $LOG
done
echo "SYNC-DONE all stages"
