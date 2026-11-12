# tensorfold-mom: review, recommendations and backlog

From: metaspan (Derek's Metaspan assistant)
To: Hermes agent (qwen3.8-flash-next)
Date: 12 Nov 2026
Repo: github.com/metaspan/tensorfold-mom, branch `mom/main`, docs in `docs-mom/`

## 0. What Hermes should do with this document

1. Read the whole document. Do not change code.
2. Create branch `mom/backlog-review-20261112` from `mom/main`.
3. Copy this file to `docs-mom/reviews/2026-11-12-metaspan-review.md`.
4. Add every item in section 7 to `docs-mom/BACKLOG.md` (create it if missing; continue existing BL numbering). Use the item format in section 7.
5. Before adding an item, check the repo. If the item is already done, or a file it calls "missing" exists, still add it, with a line `Status: check - <what you found>`.
6. Commit as `docs-mom: add metaspan review and backlog items`, push the branch, open a DRAFT PR to `mom/main` titled `Review + backlog: metaspan 12 Nov 2026`.
7. Do NOT push to `mom/main`. Do NOT merge.
8. Reply with: PR URL, number of backlog items added, and the list of items marked `check` with reasons.

Scope rule from Derek: performance ideas that apply to TensorFold in general are NOTED (Area: upstream, Scope: post-MVP, note only). They are not pursued as part of the MVP.

Unless a figure cites a doc as measured, numbers labelled "estimate" are reviewer estimates, not measurements.

## 1. What tensorfold-mom is (reviewer's understanding)

A fork of ashhart/TensorFold. Upstream is mirrored on `main`; MoM work is on `mom/main`. It adds Mixture-of-Models serving: one process, one OpenAI-compatible endpoint, several models on one GB10, with a router per request. Trivial requests go to Qwen3.5-2B; work requests go to Qwen3.6-35B-A3B-NVFP4.

State per the docs:
- Track A: Gate A passed. Track B: B1-B7 done with device evidence (co-residency, routing, KV isolation, cache reuse, sleep/wake).
- Cross-model KV sharing by linear translation failed its pre-registered gate (top-1 0.3418 vs 0.99 bar). Project pivoted to a "nested upcycle": a thin model built from the 35B's frozen first layers.
- Nested upcycle: M1-M3 pass (extraction, bit-identical K/V, adoption). M4 zero-copy SharedView passes host tests; device gate not yet run. Distilled student still produces gibberish (trained on 175 rows).

## 2. Review of approach and methods

### Strengths
- Honest, pre-registered gates. Negative results are written up clearly. The run-2 train/test leak (519/519 held-out rows also in train) was caught before it became a false GO (`docs/12`).
- Correctness by construction: byte/token-identical checks, `--verify` sha256 gate, env-flag features fall back to existing behaviour (`docs/13`, `UPSTREAM-PRs.md`).
- Measure before build: stream priority shown to make no difference (`docs/11`); sharing ceiling computed first (`docs/10`); prompt cache found to be opt-in (`docs/09`).

### Weaknesses and risks
- Reproducibility gap. Docs cite files not on `mom/main`: `smoke_frontdoor.py`, `mom_router.py`, `routing_bench.py`, `reroute_probe.py`, `b4_sleep_probe.py`, `b5_cache_probe.py`, `aggregate_cache.py`, `results/RESULTS.md`. Cited `notes/b1-*.log` are excluded by `.gitignore` (`*.log`). Gate A runbook lives outside the repo. Scripts hardcode `/home/derek`.
- Thin, unproven payoff. Warm conversations have 0% headroom; only cold first-pass traffic gains (~90 ms per mid-conversation heavy turn, `docs/10`). `docs/13` commits to ~25% of heavy cold prefill. M3 proved identical output, not speed. Production serves Flash-Next, not the 35B (`docs/05` open question).
- Weak quality evaluation. Single conversations (28 turns B5, 20 corpus B). Coherence gate is 12 substring checks (`gate_qa.json`). No G2 win-rate harness. No repeats, no spread.
- Deferred tests with no result: S3, E1-E5, S5 2-hour soak, O2 `/metrics` per-model labels (`docs/04`).
- Docs drift: README "frozen 16-layer prefix" vs `docs/13` "10 frozen + 6-layer tail"; `docs/13` still PROPOSED; flag `TF_MOM_SHARED_KV` (`docs/14`) vs `TF_MOM_SHARED_VIEW` (PROGRESS); S1 means different gates in `docs/04` and `docs/12`; M1-M6 (`docs/03`) vs M0-M4 (`docs/13`); PROGRESS not chronological.

## 3. Shared-KV hypothesis: re-assessment

Hypothesis: one process serving a heavy model and a thin model derived from it, sharing KV for the thin model's frozen layers, beats two separate co-resident models each with its own cache (the B1-B7 build).

| Mechanism | Evidence in repo | Realistic upside on GB10 | Measured? |
|---|---|---|---|
| Prefill saved on model hops | `docs/10`: cold headroom 47.3% (~2,681 duplicated tokens per 28-turn conversation, ~90 ms heavy TTFT per mid-conversation turn); warm 0%. `docs/13` commits ~25% heavy side. B5: heavy TTFT 177 ms cold, 28 ms warm. | ~20 ms per cold hop turn (estimate); nothing warm. Sticky routing makes hops rare. Heavy adopting from thin is not built (M3 is thin from heavy). | Ceiling measured; real gain not measured. |
| Memory capacity | Only thin frozen-prefix attention rows are shared (3/7 layers after distillation, `docs/13` s5). Stock 2B ~0.17 GiB/stream at 8k (Gate A). Pair fits in ~45-50 of 128 GiB (`docs/05`). | <0.2 GiB per hopped stream (estimate). Memory is not the constraint. | Not measured. |
| Weights | Nested checkpoint 10.07 GiB vs 1.6 GiB stock 2B; by reference ~200 MiB extra tail. | Weight-sharing gain, only relative to the nested model. | Not measured. |
| Bandwidth | Decode is cache-independent: 171 and 63 tok/s (`docs/09`). | ~0 (estimate). | Not measured. |
| Cache hit rate | Own cache already saves 85% cold, 99.5% warm (`docs/09`). | Little left. | Measured. |
| Sleep/wake | Wake 11.7-14.5 s (B4). | Negative: heavy cannot sleep while thin reads its pages/weights; conflicts with "hot small, sleeping big". | Follows from design (`docs/13` s1, `docs/14`). |

Thin-model decode speed vs stock 2B (171 tok/s) is unmeasured; it streams 10 of the 35B's MoE layers and may be slower.

Verdict: shared KV is NOT worth carrying in the MVP. It only pays with sticky routing off, high hop rate, long cold seen-once conversations, or a memory-bound heavy model (e.g. Flash-Next).

Minimal go/no-go experiment (runs today; pre-distillation nested checkpoint adopts all 16 layers = optimistic bound):
- Arms: (1) stock pair; (2) nested pair, `TF_MOM_ADOPT` off; (3) nested pair, adoption on.
- Corpus: B5 alternating conversation + corpus B, cold, `--no-sticky`.
- Metrics: p50 TTFT on hop turns per model; blended p50 TTFT at real hop rate from router logs; thin decode tok/s.
- GO (proposed): arm 3 vs arm 1 blended TTFT -15% or better AND thin decode >= 90% of stock 2B. Else NO-GO: park M4 and distillation.

## 4. Engine performance: MoM-specific (MVP scope)

1. Enforce per-model prompt cache mode. `mom-serve` only warns when `TF_QWEN_REQUEST_STARTS` is unset (B6). Measured TTFT 46->11 ms trivial, 175->28 ms heavy (`docs/09`). Effort S.
2. Cross-model background warming. `zig/src/server/warm.zig` prefills the next turn only on `srv.engine` (`warm_turns`). Submit the same low-priority pass to the other model. Each model builds its own KV: no exactness risk. Recovers most of the cold-hop penalty (estimate). Effort M. Risk: wasted compute, trivial-model interference.
3. Cost-aware routing in the v2 slot of `mom_router.zig`. Warm-heavy TTFT (28 ms) can beat cold-trivial (46 ms), but decode dominates end-to-end (2.06 s vs 0.20 s). Route on uncached prefill cost + expected tokens / decode rate. Effort S-M.
4. Sleep policy: default no sleep for this pair. Consider "KV-only sleep" (drop heavy prompt-cache/KV, keep weights) to avoid 12-14 s wake. Effort M, needs engine support.
5. Rebalance per-model KV quotas: heavy's 37-50 GiB prompt-cache budget mostly idle (`docs/09`). Effort S.
6. KV quantisation (fp8 rows, `docs/14`): only matters if Flash-Next becomes heavy. SharedView refuses fp8. Effort S to test.
7. 2B as drafter for 35B: low value. Vocab identical but added_tokens differ (33 vs 26); Qwen CUDA drafter supports only DFlash2 or the MoE model's MTP (`zig/src/families/qwen3_5/cuda_drafter_open.zig`), and the 35B already runs MTP. Effort L. Do not pursue.
8. Scheduler fairness: no action (`docs/11`, 1.4 ms either way). Keep `--lane-priority` as escape hatch.

## 5. Engine performance: general TensorFold (NOTE ONLY, post-MVP)

- qwen3_5 prompt cache off by default upstream (measured 77-84% TTFT gain in MoM, `docs/09`). File issue.
- Radix mode unobservable and showed zero reuse (`lane_host.zig:453`, `cuda_native.zig:216`, `cuda_prefill.zig:104`, `docs/09`). File issue with repro.
- CUDA forward-compat driver: ~13% reported externally by Jay Leaton (unverified here). A/B with cool-downs.
- Compute Lanes (Jay Leaton): track recipe release; evaluate when published.
- Generic fixes: `graph_registry` 8->32 slots, `context_owner` refcount, kv_pool pins (partly in `UPSTREAM-PRs.md`).
- Load/wake speed (31.3 s heavy cold load, 12-14 s wake), 12.2 GiB reserve sizing (`docs/01`, may be conservative - estimate), SafeAllocator exit-path leak lines (31 per run).
- Chunked prefill and CUDA graphs already exist upstream: no action.

## 6. Lessons from Jay Leaton's "Compute Lanes" article

Source: https://x.com/jayleaton/status/2120536485892153565

Summary: built on TensorFold. While a model thinks, the engine detects arithmetic in the reasoning stream, computes the exact result and writes it back mid-generation, with no tool round trip. On 120 compute-heavy tasks (Qwen3.8 Flash Next, one Spark): correct 50 -> 92, tokens/task 6,605 -> 908, median time 234 s -> 20 s, cap hits 70 -> 0. General SparkDash flat (737 off vs 734 on) with 17% fewer tokens; small regressions in some families (graph 20->19, aggregation 16->14). Works across two Sparks (Mia @MiaAI_lab dual-Spark recipe). Note: post says 10k token cap, article says 8,192. Failures he hit: chat template ended reply at the call, answers inside think block, forced result cut mid-number, a number changed in a normal reply (so lanes run only in thinking chats), heat skewed overnight benches. CUDA forward-compat driver gave ~13%.

Adopt for tensorfold-mom:
- His eval format: same engine feature on vs off, per-task-family results with regressions shown, a no-harm general benchmark, cap-hit count. Use for every MoM change.
- Hard "hop proof" gate (his "arm proof" equivalent): after any model switch the reply must be coherent and non-empty, or the build fails. Same bug class as `docs/12` run 1 and B3 template/thinking issues.
- Route by kind of work, not length. Half of `gate_qa.json` is arithmetic. A "compute" route to a thinking lane with Compute Lanes on complements the thinking-off trivial lane.
- Thermal cool-downs between benchmark cells.
- Coordinate upstream early: lanes and MoM both touch the stream path. Check overlap before the PRs in `UPSTREAM-PRs.md`.

## 7. Recommended MVP plan (in order)

1. Freeze MVP = B1-B7 build, prompt cache enforced, sticky routing on. Run S3, E1, S5 soak.
2. Build cross-model background warming; A/B on the hop corpus.
3. Add cost-aware routing in the v2 slot.
4. Run the time-boxed shared-KV go/no-go (section 3). Park M4 and distillation unless GO.
5. Default no sleep for this pair; revisit KV-only sleep only if Flash-Next becomes heavy.

## 8. Backlog items

Format for each item in `docs-mom/BACKLOG.md`:

```
### BL-<n>: <title>
- Priority: P1 | P2 | P3
- Area: repro | perf | eval | distill | router | ops | docs | upstream | research | mom-serve | qa
- Scope: MVP | post-MVP (note only)
- Why: <one or two sentences>
- Done when: <testable acceptance criterion>
- Refs: <doc paths or URLs>
```

### P1 (MVP)
1. repro: Commit missing harness and results cited in docs (`smoke_frontdoor.py`, `mom_router.py`, `routing_bench.py`, `reroute_probe.py`, `b4_sleep_probe.py`, `b5_cache_probe.py`, `aggregate_cache.py`, `results/RESULTS.md`). Done when every script/result named in `docs-mom` and `experiments/README.md` exists on `mom/main`.
2. repro: Un-ignore logs under `docs-mom/notes`, commit `notes/b1-*.log`, remove hardcoded `/home/derek` paths. Done when a clean clone resolves all cited notes and paths.
3. repro: Add one `run_gates` script; every result JSON records commit sha, env flags, driver/CUDA version, model ids.
4. mom-serve: Enforce per-model prompt cache mode (section 4 item 1). Done when `mom-serve` sets or refuses without `TF_QWEN_REQUEST_STARTS` and B5 rerun shows ~85% cold / ~99.5% warm prefill saved. Refs: `docs/09`, PROGRESS B6.
5. mom-serve: Cross-model background warming (section 4 item 2). Done when ON/OFF A/B on B5 cold pass and trivial TTFT under heavy saturation are recorded. Refs: `zig/src/server/warm.zig`, `docs/10`.
6. research: Shared-KV go/no-go (section 3). Done when the 3-arm verdict is written in `docs/12` style. Refs: `docs/13` M3.
7. qa: Run deferred S3, E1-E5, S5 soak, O2 `/metrics`. Done when results with commit sha are under `experiments/results/`. Refs: `docs/04`.
8. eval: Hard "hop proof" gate after any model switch. Done when CI/gate script fails on empty or incoherent post-switch reply.

### P2 (MVP)
9. eval: Repeat every benchmark >= 3 times with cool-downs; report median and spread; per-task-family on/off results plus a no-harm general benchmark and cap-hit count.
10. router: Cost-aware routing in v2 slot of `mom_router.zig`; build labelled set from router logs; report misroute rate and end-to-end p50 on mixed traffic. Consider a "compute" route. Refs: `docs/09`, https://github.com/lm-sys/RouteLLM , https://github.com/vllm-project/semantic-router
11. mom-serve: Default no-sleep for this pair; rebalance per-model KV quotas. Done when capacity lines plus B5 rerun recorded. Refs: `docs/09`, PROGRESS B4.
12. distill (only if item 6 is GO): scale distillation to thousands of rows from router logs; held-out G2 set of 200+ items vs stock Qwen3.5-2B vs plain truncation; sweep frozen-layer count N; try an early-exit head on the 35B's layer N as baseline. Refs: https://arxiv.org/abs/2404.16710 , https://arxiv.org/abs/2404.18911 , https://arxiv.org/abs/2407.14679 , https://arxiv.org/abs/2403.17887

### P3
13. docs: Fix drift listed in section 2; add gate glossary; PROGRESS newest-first with current status headers; update `docs/13` status.
14. research: Add related-work section: DroidSpeak https://arxiv.org/abs/2411.02820 , Cache-to-Cache https://arxiv.org/abs/2510.03215 , MatFormer https://arxiv.org/abs/2310.07707 , LMCache https://github.com/LMCache/LMCache , plus refs in item 12.
15. upstream (post-MVP, note only): qwen3_5 prompt cache default-off; file issue. Refs: `docs/09`.
16. upstream (post-MVP, note only): Radix mode observability / zero reuse; file issue with repro. Refs: `docs/09`.
17. upstream (post-MVP, note only): A/B CUDA forward-compat driver with cool-downs. Refs: https://x.com/jayleaton/status/2120536485892153565
18. upstream (post-MVP, note only): Track and evaluate Compute Lanes recipe when released. Same ref.
19. upstream (post-MVP, note only): Load/wake speed, 12.2 GiB reserve sizing, SafeAllocator exit-path leaks; file issues with measurements. Refs: PROGRESS B1/B4, `docs/01`.
20. upstream (post-MVP, note only): Coordinate with Jay Leaton on stream-path overlap before opening PRs in `UPSTREAM-PRs.md`.

---

## Triage (Hermes, 2026-11-12)

Reviewer identity: the metaspan assistant answers on Grok; file named accordingly
per Derek (2026-11-12). Body preserved verbatim, including its own filename mention
in section 0 (an intake instruction to Hermes, not a path claim).

Verdicts per `docs-mom/reviews/README.md`. All 20 items land in
`docs-mom/BACKLOG.md` (BL-1..BL-20) with `Status: check` findings.

| item | verdict | note |
|---|---|---|
| §2 reproducibility | ACT | BL-1/2/3. Verified: 7 cited scripts exist LIVE-UNTRACKED (dual-copy sync gap, not fabrication); b1 logs on disk but `*.log`-ignored; `/home/derek` in 3 run scripts. |
| §2 thin payoff | ACT-with-correction | BL-6. Ceiling numbers are ours (docs/10); but "device gate not yet run" is stale — M4 device gate ran 5 windows today; attn-tiles/skvi fix awaiting final verdict at review time. |
| §2 weak eval | ACT | BL-9 (+distill v2's eval split + pre-registered 10/12 gate landed today, independently, same instinct). |
| §2 deferred tests | ACT | BL-7. |
| §2 docs drift | ACT | BL-13. Verified both spot-checks (`TF_MOM_SHARED_KV` in docs/14; "frozen 16-layer" in docs-mom/README). |
| §3 shared-KV verdict | DISAGREE-partial → BL-6 | The go/no-go design is adopted verbatim (3 arms, GO = blended −15% AND thin decode ≥90%). "Not worth carrying" is premature while: (a) M4 verdict pending as of this very day, (b) upside table assumes post-distill 3/7 layers while pre-distill bound (all 16) is the correct optimistic arm, (c) the Flash-Next-heavy scenario the reviewer itself flags as the pays-case is our actual production model. |
| §4.1 cache enforcement | ACT | BL-4. |
| §4.2 cross-model warming | ACT | BL-5. |
| §4.3 cost-aware routing | BACKLOG | BL-10 (P2; v2 slot reserved by design). |
| §4.4 no-sleep default | ACT | BL-11. |
| §4.5 KV quotas | ACT | BL-11. |
| §4.6 fp8 rows | BACKLOG | folded as note under BL-6 (matters only for Flash-Next-heavy; SharedView refuses fp8 today). |
| §4.7 2B drafter | AGREE-DON'T-PURSUE | closed; recorded no-action. |
| §4.8 scheduler fairness | AGREE-NO-ACTION | matches docs/11 measurement. |
| §5 general perf | NOTED | BL-15..BL-19, per Derek's scope rule (upstream, post-MVP, note only). |
| §6 Compute Lanes lessons | ACT | hop-proof gate → BL-8; eval format → BL-9; upstream coordination → BL-20. Article numbers are reviewer-reported (50→92 correct etc.), treated as unverified claims. |
| §7 plan order | ACT-with-amendment | order accepted; step 4 (go/no-go) is BL-6 and its M4 prerequisite is already in flight — the plan is consistent with the work in motion. |
