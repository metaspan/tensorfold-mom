# spark4 handoff — GPU-training migration for the TensorFold MoM nested program

Date: 2026-11-16 | From: Hermes on spark-2d3b (spark1) | To: Hermes on spark4
Owner: Derek. Read this fully before touching GPU.

## Mission split (agreed with Derek)

- **spark4**: all RAM-hungry HF-side training — `distill_train.py` (student t-suite),
  `mtp_train.py` (MTP head), `distill_kd_prep.py` (teacher-logit prep), `teacher_gate.py`,
  `distill_gate.py --engine hf`, `d1_trainset.py`, GKD student-rollout collection.
- **spark-2d3b stays authoritative** for: native engine everything (NVFP4 boots,
  `distill_gate --engine native`, decode/tok/s probes, SharedView/adoption windows),
  production :8000 (LIVE — do not disturb from here), and quantisation of exported
  students via the NVFP4 path.
- Checkpoints trained on spark4 get rsynced back to spark-2d3b `~/models/tensorfold-mom/`
  for native gates/probes. HF gates (12-item, bar >=10/12, `gate_qa.json`) may run on spark4.

## Repo

`git clone https://github.com/metaspan/tensorfold-mom` branch **mom/main** (NOT main —
main is a pure upstream mirror; MoM content only exists on mom/main).
Everything you need lives in `src/tensorfold/experiments/` (trainers, gates, harnesses)
and `docs-mom/` (BACKLOG = the book of record: pre-registrations, adjudications, kill criteria).

## Program state (2026-11-16, condensed — BACKLOG.md is authoritative)

- M3 KV-adoption gate CLOSED; M4 SharedView root-caused & fixed (dedicated skvi ids buffer);
  M1 GDN head-dup done. BL-6/6T decode NO-GO (141.4 < 154 bar, not config-addressable).
- Student distillation t1–t4 ALL FAIL the 12-item gate (0,0,3,3,4/12). D1 (train-row
  underfit) + D2 (layer-10 early-exit 0/12) + 5e (last-6 byte-copy gate 1/12, byte-verified,
  reproduced) = every zero-training reconstruction fails.
- **CORRECTED model of the failure** (checkpoint-beats-spec, 2026-11-15): our tail layers DO
  have full routed experts (256/layer, fused tensors on disk); the real defect is
  DEPTH-POSITION MISMATCH — t1–t4 trained parent layers 10–15 as the final layers behind an
  lm_head calibrated for the post-39 stream. Reviewer "shared-expert-only tail" framings are VOID.
- **Next rung 5g (HOLD — awaiting Derek's GO)**: same trainer/corpus/gate as t1–t4, but the
  tail warm-starts from parent layers **34–39**. The bf16 twin is BUILT:
  `qwen36-nested-16l-last6-bf16` (asset list below). ~4–5 GPU-h. BL-21 kill budget ~16/20 GPU-h.
- BL-21 MTP head: trainer now uses `torch._grouped_mm` (18x faster, boot parity self-check);
  the decisive measured tok/s-vs-154 probe run keeps getting killed by unified-RAM OOM on
  spark-2d3b — that's exactly what spark4 is for. Re-run `mtp_train.py` (see phase2.sh P2/P3),
  then ship `mtphead-t3` + merged `qwen36-nested-16l-t3-mtp` back for the native probe.

## Assets to ship spark1 -> spark4 (rsync; totals ~250 GB)

Under `/home/derek/models/tensorfold-mom/`:
- `qwen36-35b-a3b-fp8-deq/` (67 G dequantised bf16 teacher — validated: teacher-gate 11/12)
- `qwen36-nested-16l-bf16/` (27 G twin/warm-start), `qwen36-nested-16l-last6-bf16/` (27 G 5g twin)
- `qwen36-nested-16l-t1..t4/` (27 G each — students for analysis/GKD rollouts; t3+t4 mandatory)
Under `/home/derek/tensorfold-mom/src/tensorfold/experiments/`:
- `results/distill/rollouts-10k-clean.jsonl`, `results/distill/kd-10k.pt` (12,226 rows)
- `results/night/` ledgers + `teacher-gate*.json`, `d1-t3.json` (forensic record)
- `gate_qa.json` + all harness scripts (in git).
SKIP: `qwen36-35b-a3b-nvfp4/` (native path stays on spark-2d3b; spark4 cannot boot it —
the native engine refuses bf16 MoE AND spark4 has no zig/fatbin build).

## Stack pins (exact; `docs-mom/reviews/requirements-kd-stack.lock` is the manifest)

- venv python 3.12; transformers 5.8.1; **kernels==0.12.3** and huggingface-hub>=1.5,<2.0
  (NEVER bare `pip install -U kernels` — 0.18 breaks hub_kernels import; 0.6–0.11 force hub<1.0);
  fla 0.6.0; **do NOT install causal_conv1d** (breaks GDN path); torch >= 2.11 (needs `_grouped_mm`).
- HF eager is NOT a usable GDN reference; do not chase HF-vs-engine logit diffs.

## Operating rules (non-negotiable)

1. **Pre-register before training**: gate thresholds + config fixed in BACKLOG before any
   GPU spend; one registered run per config (no config fishing). Amendments harness-only, dated.
2. Kill criteria are real: BL-21 = t4 + one ladder step + 20 GPU-h total, then re-scope.
   Log GPU-hours per run into the ledger (spark4 section of results/night/).
3. Night/long queues: single GPU-serial orchestrator, `systemd-run --user --scope -p
   MemoryMax=<cap>` per heavy job, logs + status ledger OUTSIDE /tmp, per-stage `timeout`,
   hourly HEARTBEAT lines. spark-2d3b lost one night to a hard crash + /tmp wipe — kernel
   7.0.0-1024 got installed by unattended-upgrades mid-run; consider `apt-mark hold` on
   kernels on spark4 too.
4. Coherence gate = 12 items, bar >=10/12, `distill_gate.py --engine hf` for bf16 students
   (native REFUSES bf16 MoE categorically — cuda_geometry.zig:290). Teacher calibration
   gate uses max_new_tokens=128 (budget amendment is pre-registered; the bar stays 11/12).
5. Never two concurrent >20 GiB jobs on one unified-RAM box.
6. MoE compute paths in trainers: sort-based grouped experts or `torch._grouped_mm`; any
   compute-path change ships with boot-time parity self-check + 10-step smoke (this caught
   3 shipping bugs for minutes of cost).
7. pkill patterns: bracket-trick (`mtp_trai[n].py`) or you kill your own shell.
8. Verify pushes: `gh api repos/metaspan/tensorfold-mom/git/ref/heads/mom/main --jq .object.sha`
   (silent push failures happened for a whole day once).
9. Production :8000 on spark-2d3b is LIVE — nothing spark4 does may touch it.
10. Reports to Derek: numbers with artifact paths behind them; no claim without a run file.

## Immediate queue for spark4 (once assets land; Derek green-lights each ladder step)

1. Stack bring-up + smoke: `test_moe_parity.py`, then `mtp_train.py --smoke` (expect
   `MoE grouped-vs-reference maxdiff ~1e-3` and 10 clean steps).
2. `mtp_train.py` full on t3 (batch 16; drop to 8 if spark4 RAM <100 GiB) -> ship head +
   merged dir back for the native tok/s probe (that number decides BL-21).
3. 5g training run (ONLY with Derek's explicit GO): `MOM_STUDENT=<last6-bf16 twin>
   distill_train.py --data results/distill/rollouts-10k-clean.jsonl --epochs 3 --seed 7`
   -> export -> HF gate on spark4 -> ship checkpoint back for native-side confirmation.
4. If 5g fails the gate: GKD rung 5f (student-rollout corpus via distill_collect variant,
   then kd-prep on those rows, then train) — pre-registration already in BACKLOG, needs
   Derek's awareness too.
