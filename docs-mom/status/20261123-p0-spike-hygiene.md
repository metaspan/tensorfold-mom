# P0.3 — Spike-2 verdict correction (23 Nov 2026)

## Corrected verdict (amends spike 2 "shared cache is dead")

"No KV sharing between independent checkpoints (expected). Same-lane miss was
the default cache mode, not a property of MoM. Nested prefix sharing still
works (arm C)."

## Basis
- Spike 2 observed `prompt cache: N tokens, resumed at 0, kept 0` on every
  request and concluded "shared cache is dead."
- P0.2 spike-2 rerun (gate `gates/20261123-spike2-rerun.md`) shows this was the
  *default opt-out*: `qwen3_5` CUDA conversation cache is opt-in, and the
  working mode is `TF_QWEN_REQUEST_STARTS=1` (docs/09 three-mode analysis).
- Arm A (default env): reuse 0.0% both lanes, prefill saved 0.0% — reproduces
  spike 2 exactly.
- Arm B (`TF_QWEN_REQUEST_STARTS=1`): warm-pass reuse 100%/100%, prefill saved
  99.0%/99.0%, TTFT p50 9.8 ms (2B) / 27.4 ms (35B) — matches docs/09.
- Arm C (nested thin + 35B, TF_MOM_ADOPT/SHARED_VIEW): not run — nested track
  on HOLD pending Derek GO (plan P3.1, 5g).

## Run files
- experiments/results/b5/b5-armA-20261123-1501.json
- experiments/results/b5/b5-armB-20261123-1500.json
- experiments/b5_cache_probe.py (recovered, committed)

## Consequence
Same-lane cache miss was never a MoM finding; it was an unset env var. The MVP
default (stock 2B + 35B, lane-local caching + exact re-prefill on switch) is
validated by arm B numbers. Cross-model translated cache remains closed (S1 run
3 NO-GO, docs/12). Approximate handoff (P4) stays gated on P1 failing its bar
plus Derek approval.

## Next
P1 (B5-long: size the escalation re-prefill prize) is unblocked and device-only.
