# BL-6T: Tuned thin-decode probe — PRE-REGISTERED 2026-11-13 (before any run)

## Why
BL-6 round 3 (results/bl6/bl6-20261113-0904.json) gave the first valid measurement:
TTFT PASS (copy 0.405 / view 0.416 ≤ 0.85), decode FAIL (A_stock mean 141.9 tok/s vs
pre-registered ≥154 = 0.90× the production parent's 171). Decode was uniform across
arms (141.9–145.9, mean by arm within 1.4 tok/s) ⇒ not adoption-attributable. This
probe tests the remaining alternative: is ~142 a *config* ceiling of the untuned
serve line (`--parallel 2`, default context 32768) or intrinsic?

## Legality of the knobs (audited against the build @ 2a2093257)
- `--kv-cache fp8`: REFUSED for this family — `cuda_decide.zig` returns
  `error.QwenBf16Kv8` when asked. NOT a knob.
- MTP/drafter: thin checkpoint has no `mtp-4bit.safetensors` (serve line:
  `drafts: copies only`); no cached drafter asset exists for this checkpoint. NOT a knob.
- CUDA graphs: engine captures decode shapes automatically (`e.graphs`,
  cuda_engine.zig ~497–509), not a toggle. NOT a knob.
- Legitimate, available: `--parallel 4` (stream slots; parent-lane co-residency
  contention), `--context 8192` (shrinks the KV arena; 128-token prompts — smaller
  arena may improve locality), `TF_MOM_TRIVIAL_PRIORITY=1` (stream priority,
  cuda_engine.zig:222).

## Scenario D_tuned (exactly one difference from A_stock: the boot line)
Boot: `mom-serve --host 127.0.0.1 --port <gate port> --parallel 4 --no-sticky
--model <NESTED> --name <NESTED_NAME> --lane trivial --pin
--model <BIG> --name <BIG_NAME> --lane default` **plus `--context 8192` on the
nested spec if accepted, else global**, env adds `TF_MOM_TRIVIAL_PRIORITY=1` and
`TF_QWEN_REQUEST_STARTS=1`. Same parent_replies.json variants 0..2, same 96-token
greedy streaming turns as bl6_gate A_stock. Harness: experiments/bl6t_probe.py.

## Thresholds (FIXED NOW, will not move)
- PRIMARY: mean tok_s over the 3 variants ≥ 154.0 (same bar as BL-6 decode).
- SIDE GUARD: every variant's tok_s ≥ 138.0 (no per-variant collapse).
- Any run that fails the primary is a measured NO-GO for the tuned arm; BL-6's
  verdict is NOT re-adjudicated by this probe (different scenario).

## Consequence (decided now)
- PASS ⇒ a tuned config exists that clears the decode bar; M4 performance story
  = "tuned thin" and a follow-up BL-6 round re-measures B/C on the tuned line.
- FAIL ⇒ the ~142 tok/s ceiling is intrinsic to thin bf16 decode on this hardware
  without a drafter; the decode bar can only be met by NEW ASSETS (an MTP head for
  the nested checkpoint, or engine decode optimizations). No further config
  variants may be run "looking for a pass" — that would be post-hoc fishing.

## Result
(to be appended verbatim after the single pre-registered run)

**Run 2026-11-13 09:16 UTC (results/bl6t/bl6t-20261113-0916.json): NO-GO.**
mean 141.4 tok/s (bar ≥154), min variant 141.2 (guard ≥138 passed), 96 tokens ×
3 variants, boot line verified `--parallel 4 --context 8192` + trivial stream
priority. Per the pre-registered consequence: the ~142 tok/s ceiling is INTRINSIC
to thin bf16 decode on this hardware without a drafter; no further config variants
may be run looking for a pass. The decode bar can only be met by NEW ASSETS —
an MTP/drafter head for the nested checkpoint, or engine decode optimizations.
Feeds BL-21: thin decode 141.4 < Qwen3.5-2B's 171 ⇒ (3) says no routing case on
decode alone; the routing case must stand on the adoption TTFT (0.046 s vs 0.11 s
full prefill + parent hop) or be re-scoped — programme decision for the owner.
