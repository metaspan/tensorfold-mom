# P4 handoff spike gate — 20261124 (approximate attention-KV handoff)

## Pre-registration status
Derek approved starting P4 on 2026-11-23 (P1 bar missed: heavy delta-prefill
p90 3277 ms, P4 precondition met). This file is written BEFORE any P4
experiment runs (plan P4.1).

## Question
Does translating the thin lane's attention K/V into the heavy lane's
attention space (512→512 linear maps on the 6 shared full-attention pairs,
offline-fit — docs/12 geometry: attention-favorable, GDN unfavorable so GDN
state is recomputed) let the heavy lane answer escalation turns bit-same as
its exact prefill — turning multi-k escalations from ~3.3 s p90 (P1 run-01)
into cache-hit TTFT?

## Scope honesty (S1 context)
S1 run 3 already measured ROW-level translation between these independently
trained checkpoints as decision-grade NO (top-1 0.3418 on novel rows). P4
re-tests at the only bar that matters operationally: END-TO-END greedy
agreement of the heavy lane's continuation, with the engine's exact-verify
gate conceptually in front (translated KV is never a truth source, docs/12).
A P4 PASS means agreement is achievable despite row error (the continuation
is robust); a FAIL (expected under S1) closes cross-model translation fully.

## Pre-registered metric + bar
- Harness: `experiments/p4_handoff_fidelity.py` (offline, HF-side; native
  refuses bf16 MoE so the engine-side reuse test is P4.3 only on a PASS).
- 20 escalation turns: conversation absorbed on the thin lane (stock 2B
  serving, B5 corpus + B5-long blocks), then the heavy lane answers the
  escalation prompt.
- Baseline: heavy lane (HF, fp8->bf16 dequant teacher weights, greedy,
  thinking off) prefills the exact full prompt; record greedy continuation
  (32 tokens).
- Treated: heavy lane receives translated attention K/V for the prefix via
  the injected-cache path, attention layers read injected rows (never
  recomputed), GDN layers run normally over the prefix tokens (their state
  is NOT translated); greedy continuation from the escalation prompt.
- BAR: greedy-token agreement with baseline = 20/20 turns (every continuation
  identical token-for-token). ANY miss -> STOP: P4 dies, no P4.3 engine work.
- Fitting rule: maps fit ONLY on a fit set (8 conversations); the 20 eval
  escalations are novel rows (S1 run-3 memorization lesson applies: a
  same-conversation holdout is NOT decision-grade).
- Secondary recording (not bar): prefill-token reduction estimate, map
  conditioning, per-layer residual.

## Cost
Fit + eval offline on HF models; heavy loads once (~66 GiB bf16 from the
fp8 teacher). ~1-2 h device after the 5g retrain window (single-GPU rule).

## Run files
- experiments/results/p4/run-NN/ (env.txt, build.txt, maps.json, per-turn
  agreement table, raw generations)

## Verdict rule
- 20/20 -> schedule P4.3 (2-3 day spike, 300-500 lines: engine export hook +
  map + verify-gated load; discard if <2x faster than exact re-prefill).
- < 20/20 -> record, discard, and cross-model translation is closed for good;
  MVP (P2) is the serving config; nested track carries the cache prize.
