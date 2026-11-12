# Test plan

Principle: TensorFold's exactness contract is *per-path, drafted-vs-plain and
concurrent-vs-solo at identical settings* (see recipe `tools/exact.py`,
`tools/prompt_reuse.py`, `tools/toolcheck.py`). MoM inherits it by construction — each
lane runs an unmodified pipeline — so tests focus on (a) the new routing/scheduling
surface, (b) boundaries where lanes could touch (memory, caches, shared CUDA globals),
(c) the brief's v1 success criteria as formal gates.

Tiers: T0 host (no GPU), T1 single-engine GPU sanity, T2 MoM integration, T3
qualification/perf. Every Track-B phase lands with its T0 tests green; T2/T3 run on
spark-2d3b outside production windows (production :8000 never co-scheduled — see skill
hard rules).

## 1. Router (T0, host-only; Track A shares the same policy spec)

| # | Test | Pass criteria |
|---|---|---|
| R1 | Policy table unit tests: explicit id wins; unknown id → 400 (chat path!) ; trivial rules (length < N, pattern list, tools/images force big); sticky on/off; default fallback | 100% branch table covered; decision + reason exact |
| R2 | Edge cases: empty messages; unicode/mojibake length; system-only prompt; 0 max_tokens; batch `n>1`; streaming vs non-streaming same decision | deterministic, documented behavior |
| R3 | Decision latency microbench (10k synthetic bodies, incl. 100k-char prompt) | rule path p99 < 1 ms (budget 5 ms includes IPC only in Track A) |
| R4 | Sticky LRU: capacity, eviction, conv-key derivation fallback (system+first-user hash), free re-route when disabled | exact re-route matrix |
| R5 | Config: malformed routing.json refuses startup naming the field; hot-reload is atomic (no torn rules) | refuse-over-degrade preserved |
| R6 | Concurrency fuzz: 64 threads deciding simultaneously; no shared mutable state | TSan/`-fsanitize` clean |

## 2. Registry & loader (T1/T2)

| # | Test | Pass criteria |
|---|---|---|
| L1 | N=1 byte-compatibility: MoM build single-model == upstream build on upstream test suite + receipts | zero behavior delta |
| L2 | Two-model startup: capacity lines per model; `/v1/models` lists both ids with each engine's real `info` (compare against each served alone) | exact match per model |
| L3 | Plan-time refusal: over-budget pair (fake inflated weights or lowered cap) → startup refuses, names the budget term; never silently shrinks context/parallel/top-k or disables drafts | refusal line parseable |
| L4 | Load-order swap (small-first vs big-first) same steady state | both fine; report load times |
| L5 | Park/wake (B5): park big lane → trivial lane unaffected (R-runs keep passing); wake latency < 60 s cold page cache, < 15 s warm (measured and logged) | thresholds met; pinned lane never parks |
| L6 | TP model in MoM startup → refused by name | v1 contract |

## 3. Exactness per lane (T2; reuse upstream tools per model)

| # | Test | Pass criteria |
|---|---|---|
| E1 | Per model on the MoM build: `tools/exact.py --pad 3000` equivalent (drafted greedy == `draft:false`; concurrent == solo) | token-array equality, full length not prefix |
| E2 | Sampled exactness with prompt-keyed seeds (`--prompt-seed`) per model | equal across runs |
| E3 | Tool-call conformance per model (`toolcheck`-style 3/3 + streaming variants) | all pass per lane |
| E4 | Grammar/structured output per lane (xgrammar path both lanes) | valid + identical to single-model |
| E5 | Cancellation mid-stream on lane A while lane B generates | both behave exactly as solo-under-cancel |

## 4. KV & prefix-cache independence (T2 — the highest-risk area)

| # | Test | Pass criteria |
|---|---|---|
| K1 | **No cross-model cache hits:** identical long prompt to both lanes; per-model hit counters must show independent hits only (inject distinct first-token probe? no — use counters + `prompt_reuse` analogue per lane) | lane-A reuse never credited to B |
| K2 | Re-route probe (A3 on native): conversation turns alternated forced-model at temp 0; each reply equals the solo ground truth of the answering model | exact token equality per turn |
| K3 | Prefix-cache correctness within a lane after re-route round-trips (A→B→A: A's resumed reply == A's continuous one) | equality; measures A's prefix hit (should hit its own) |
| K4 | Tag-fold negative test: forge a Store entry under wrong model tag → RAM adopt must reject (identity guard) | error, not silent adoption |
| K5 | Disk tiers per model: two compat dirs, separate flocks, restart recovery (scan drops foreign), LRU trim per-model budgets | clean separation |
| K6 | Memory-pressure path: exhaust lane-A KV quota → lane A returns `MemoryPressure` wait; lane B keeps serving | no cross-lane starvation |

## 5. Scheduler & interference (T3)

| # | Test | Pass criteria |
|---|---|---|
| S1 | Trivial-lane TTFT idle-baseline: small model dedicated (Track A single serve) vs MoM trivial route, same workload | Δ ≤ 10% (brief: "indistinguishable") |
| S2 | Interference: trivial TTFT/e2e with big lane 1×, 4×, 5× saturated decode + big-lane chunked prefill of 32k | p95 degradation ≤ agreed threshold (start 20%; stream-priority fallback below it) |
| S3 | Big-lane preserved: recipe sparkDash decode/prefill/TTFT matrix on MoM == single-model baseline (B0 numbers) | within run-to-run noise (±15% single-stream prose spread known upstream; aggregate ±5%) |
| S4 | Admission bookkeeping: `live.connections`/`live.waiting` merged view == sum of lanes under load; drain (503 + Retry-After) covers both lanes | exact |
| S5 | Soak: 2 h mixed workload, 20 rps mix 70/30 trivial/real | no leaks (RSS plateau), no stuck streams, decision log complete |

## 6. Observability (T2)

| # | Test | Pass criteria |
|---|---|---|
| O1 | Decision metrics count == harness-sent requests (per model, per reason) | exact |
| O2 | Per-model TTFT/tok-s/cache-hit/queue-depth present on `/metrics` with `model` labels | scrape + assert shape |
| O3 | Route-decision header triple present on every response; `x-tf-route-us` p99 < 5 ms | exact |
| O4 | Park/wake switch-latency histogram emitted | present |

## 7. v1 Qualification (Phase C) — brief success criteria, in order

1. Single endpoint serves both models → L2 + E1–E4 green.
2. Trivial latency indistinguishable from dedicated small server → S1 ≤10%, S2 within
   threshold, S5 soak clean.
3. Big model keeps normal TensorFold performance → S3 + E1–E3 on big lane.
4. Free re-routing without state corruption → K1–K4 green (temperature 0 exactness).
5. Memory accounting prevents OOM → L3 + K6 + S5 (dmesg clean, engine reserve intact).

Sign-off = all numbered gates pass on the native build with the `05` model pair; each
result recorded with date, commit hash, and raw logs under `experiments/results/`.

## Track A note

Track A runs the T0 router suite (shared spec), K1–K3 (front-door level: cache
independence is per-process there — trivially true — and K2 is the real signal), and
S1/S2/S5 against child processes. Track A results feed Gate A in `03` and become the
regression baseline for Track B.
