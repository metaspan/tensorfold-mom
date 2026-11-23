# Spike-2 rerun gate — 20261123

## Question
Does the B5 cache measurement (docs/09) reproduce, and does the same-lane
cache opt-in (`TF_QWEN_REQUEST_STARTS=1`) deliver the reuse the deep-dive
diagnosis predicts? Re-run of spike 2, three arms, one run each.

## Arms
| Arm | Lanes | Env | Expected (deep-dive diagnosis) |
|-----|-------|-----|--------------------------------|
| A (repro) | stock 2B + 35B | default (no opt-in) | `resumed at 0, kept 0` on every request — zero prefix reuse |
| B (opt-in) | stock 2B + 35B | `TF_QWEN_REQUEST_STARTS=1` | same-lane repeat: `resumed at` > 0; reuse >= 85% cold, >= 99% warm; tiny->big switch still a full prefill on the big lane |
| C (nested) | nested thin + 35B | `TF_QWEN_REQUEST_STARTS=1 TF_MOM_ADOPT=1 TF_MOM_SHARED_VIEW=1 TF_MOM_SHARED_VERIFY=1` | tiny->big switch logs adoption; byte-identical to copy path (BL-6 arm C) |

## Metric (per request)
`resumed at` / `kept` from the engine's `prompt cache:` line; prefill time;
TTFT; end-to-end time; token counts.

## Pre-registered pass bar
- A reproduces: reuse == 0% on both lanes.
- B: same-lane warm-pass reuse >= 99% on both lanes; prefill saved >= 85% cold and >= 99% warm.
- C: adoption logged, zero verify mismatches. **(Not run — nested track on HOLD; 5g needs Derek GO per plan P3.1.)**

## GPU-hour cap
~1 h device (two 28-turn passes x 2 arms on GB10).

## Run files
- experiments/results/b5/b5-armA-20261123-1501.json (arm A)
- experiments/results/b5/b5-armB-20261123-1500.json (arm B)
- experiments/b5_cache_probe.py (probe script, recovered/committed)

## Result (23 Nov 2026)
- Arm A: PASS — reuse 0.0% both lanes, prefill saved 0.0%, TTFT p50 44 ms (2B) / 164 ms (35B). Default = full prefill every turn, confirming spike 2's "resumed at 0, kept 0" observation was the *default opt-out*, not a property of MoM.
- Arm B: PASS — warm-pass reuse 100%/100%, prefill saved 99.0%/99.0%, TTFT p50 9.8 ms / 27.4 ms. Matches docs/09 (85.4%/85.7% cold, 99.5% warm) within measurement noise. `TF_QWEN_REQUEST_STARTS=1` is the working opt-in for this family.
- Arm C: not run (nested track on HOLD).

## Verdict
Spike 2 measurement confirmed: same-lane miss was the default cache mode
(opt-out), not a MoM property. The opt-in (`TF_QWEN_REQUEST_STARTS=1`)
restores per-lane prefix reuse. Spike 2 verdict amended — see P0.3.

Consequence: nothing blocks P1/P2 on the measurement. P3 (nested, arm C)
awaits Derek GO.
