# PROGRESS

## 2026-11-11 — B4 SLEEP/WAKE DEVICE PASS (b4_sleep_probe.py, ALL PASS)

- Two full cycles at --lane-idle 15: heavy lane flips to state=asleep in
  /v1/models, engine memory key disappears, MemAvailable rises ~23 GiB
  (GB10 unified memory; nvidia-smi memory.used is N/A here — use lane
  mom.memory + MemAvailable), trivial lane stays hot (<3 s) throughout,
  wake-on-route logs 11.7-14.0 s per wake (21.8 GiB NVFP4 reload), and
  post-wake trivial turns are byte-identical to pre-sleep baselines.
- Leak gate follows the known-red policy: mom-serve's exit path prints
  ~31 SafeAllocator lines with empty stack traces in every baseline
  (B1/B2 S2 too); failure only if they appear mid-serving (none did).
- Commits: 38a2747 (wake+lease), 70f786c (timer/flags),
  b2bc1c9 (per-lane memory). BRIEF IMPLEMENTATION ORDER 1-5 COMPLETE
  with device evidence; only item 6 (shared prefill / KV translation,
  explicitly later) and the v2 classifier slot remain.

## 2026-11-11 (early) — B4 started: docs/08 work spec + chunk 1 (a187ae7)

- docs/08-b4-sleep-wake.md: per-lane lifecycle generalized from
  idle_unload.Manager (lease/cancel semantics), policy (trivial lane never
  sleeps; --lane-idle SECONDS on non-trivial lanes), wake-on-route parks the
  caller and reopens synchronously (switch-latency logged), /v1/models gains
  "state":"asleep", invariants incl custody cross-check + post-wake
  byte-identity, device probe plan b4_sleep_probe.py.
- Chunk 1 landed: mom_sleep.zig LaneLifecycle (pure, injected clock) with 4
  host tests; test-mom-host filter 'sleep:'; green first run.
- Chunk 2 next: registry fields (last_used/state/lease/reopen), lease
  begin/end in chat submit path, wake-on-route in route().

## 2026-11-11 — REBASED onto upstream 1.2.7 (93a2e93), all gates green

- 8 MoM commits rebased onto Release 1.2.7 (137 upstream commits) with ZERO
  conflicts; backup branch mom/main-pre-rebase kept.
- One fork-attributable regression found by pristine pair-compare (baseline
  worktree pinned to the same sha): upstream's new models_idle_audit_test
  builds Server=undefined; status_routes.models() reads srv.mom -> poisoned
  segfault. Fix: init srv.mom=null in that test (commit 230f118).
- Full suite: 8914/8984 pass, 69 skip; test-cuda-host green post-fix (zig
  build retries GPU DeviceLost flakes once; pristine passed same step).
- DEVICE SMOKE on 1.2.7 binaries (proc_cad65cf64cd7): S1 PASS, S2 PASS,
  routing 4/4, SMOKE_RC=0. Production restored healthy.
- Standing rule (skill): after each rebase grep new upstream
  "Server = undefined" tests for fields the MoM paths read.

## 2026-11-11 — B6 native router latency + cache-mode guard (commit b803699)

- mom_registry.route() now times the pure decision path; /v1/models exposes
  decision_us_mean/max (lane-0 entry). Device bench
  (experiments/b6_router_bench.py, proc_5d2b0b890bd1, 85 decisions):
  **mean 4 us, max 135 us** — brief's <5 ms budget met with ~37x headroom.
  Client round-trip for a trivial max_tokens=1 request: p50 16 ms.
- mom_main warns at startup for every qwen3_5 lane running without a
  conversation cache switch (B5 finding; bench log confirmed warning fires
  when unset and stays silent under TF_QWEN_REQUEST_STARTS=1).
- Bench gotcha: sending an unknown explicit "model" id refuses BEFORE the
  route counters increment — omit the field for content-routed benches.

## 2026-11-11 — Component 3 closed + B4 soak + item-6 ceiling + classifier slot (docs/10, docs/11)

Four queued follow-ups done in one round:

1. **GPU stream priority (component 3 leftover) — measured and CLOSED.**
   `mom-serve --lane-priority` opens the trivial lane's CUDA stream at
   `.highest` (pure `priorityIndex` + setenv window around that one open;
   commit `c1de3cc`). B7 A/B, trivial TTFT p50 under 8-way 35B saturation:
   **OFF 1.4 ms = ON 1.4 ms** (p95 1.8 ms; heavy cost +2%, noise). Verdict:
   per-lane batching already delivers the fairness; priority ships inert as
   an escape hatch. docs/11-stream-priority.md, B7_RC=0.
2. **B4 5-cycle sleep/wake soak — ALL PASS** (post race-fix binary).
   Per cycle: heavy asleep, memory key absent, MemAvailable +23.4-23.5 GiB;
   trivial hot; post-wake byte-identity 5/5. Wake latencies 11923-14513 ms,
   **drift ratio 1.22** (< 2.0 gate): no fragmentation trend across 5 cycles.
   31 exit-path SafeAllocator lines, mid-serving false (known-red). B4_RC=0.
3. **Item-6 duplication ceiling — analysed** (docs/10). Warm conversations:
   **0% headroom** for cross-model KV translation (lane-local caches already
   99.5%). Cold first-pass: 47.3% of prefill (~90 ms heavy TTFT/mid-turn).
   Deprioritized: route affinity + prompt-tier text caching dominate cheaper.
4. **Classifier v2 slot — implemented, host-verified** (`cb5011e`).
   Consulted only at the rules' default branch; escalate-or-hold; explicit/
   sticky/empty-text bypass; null default = byte-identical v1. Real
   classifier model stays a later enhancement.

**Real bug caught and fixed by the soak's first attempt** (`6220749`):
lease/park race — `lease()` cancelled an already-claimed park while the timer
closed the engine regardless -> first heavy chat 503 "engine is unavailable".
Earlier 2-cycle passes had won this race by timing luck. Fix: timer holds the
lifecycle lock across close+null+finishUnloadLocked (teardown atomic);
lease() waits out unloading/loading and returns false so noteRoute retries
via wake-on-route. Invariant restored: resident => engine non-null.

Pitfalls banked: window scripts need `set -o pipefail` (a tail-masked build
failure ran probes on a stale binary once); this std has `getenv` but not
`setenv` (libc extern + void-wrapper for defer, c_int-in-defer refused).

## 2026-11-11 — B5 CACHE EFFECTIVENESS DEVICE PASS (docs/09)

28-turn alternating trivial/work conversation x2 passes, both lanes resident
(no sleep), --no-sticky free re-routing, greedy, streamed. Key finding:
qwen3_5 CUDA conversation cache is opt-in; working mode
**TF_QWEN_REQUEST_STARTS=1** (three-modes analysis in docs/09).
- Routing 28/28 per pass, 0 failures; both lanes resident whole run.
- Prefill saved: cold pass 85.4%/85.7% (trivial/work), warm 99.5% both.
- TTFT cold->warm: trivial 46->11 ms (77%), work 175->28 ms (84%).
- Decode: 171 tok/s (2B), 63 tok/s (35B NVFP4+MTP), cache-independent.
- Engine proof: resumed 2708/2715 tokens late-conversation, hits 27/lane.
- experiments/b5_cache_probe.py + aggregate_cache.py; proc_7a1b639df892;
  production restored healthy.

## 2026-11-10 (component 3 analysis) — per-model queues: design position

- Each mom-serve lane is a full engine instance with its OWN lane host +
  batching loop (core/lanes), so "per-model queues" hold structurally: a
  request routed to the trivial lane is never queued behind heavy-lane work;
  continuous batching stays model-local, exactly as the brief mandates
  ("chunked prefill and in-flight batching remain model-local").
- "Trivial queue higher priority" at the GPU layer = stream priority: the hook
  exists (cuda/stream.zig initPriority(.highest), currently no callers).
  Deferred to B4: thread a priority choice from native.open options into the
  lane host's stream creation, set highest for the trivial lane; needs its
  own device window + contention measurement (trivial TTFT under 35B load).
- Next durable items: B4 sleep/wake (reuse idle_unload hot-swap machinery),
  stream-priority experiment, B5 classifier slot (v2).

## 2026-11-10 (late night, latest) — metrics device PASS; B3 done except queue priority

- mom_metrics_check.py: routed trivial->qwen3.5-2b / heavy->qwen3.6-35b;
  /v1/models reported {"decisions":1,"role":"trivial"} and
  {"decisions":1,"role":"default"}. METRICS CHECK: PASS; production restored.

## 2026-11-10 (late night, later) — B3 per-lane metrics (commit 86e385c)

- Registry counts routed decisions per lane (mutex-guarded); /v1/models gains
  {"mom": {"decisions": N, "role": "trivial|default|other"}} per advertised id.
  Zero new routes; OpenAI-compatible extra field. Host test green; device
  verification via experiments/mom_metrics_check.py in window (see below).
- Custom-Zig style/API notes hit again: array default must be `@splat(0)`
  (`** 16` trips the fork's whitespace style checker); `for (slice) |*slot|`
  binds a POINTER here (use `slot.*`), unlike by-value captures.
- B3 now: probe PASS + metrics. Remaining brief items: queue priority in the
  shared batch loop (deepest), B4 sleep/wake + orchestrator, B5 classifier slot.

## 2026-11-10 (late night) — B3 PROBE PASS (reroute + KV isolation proven)

- b3_reroute_probe.py (3 servers, one window): dedicated-small baseline
  (non-empty greedy replies, aborts if empty); K3 same-lane continuity 3/3
  token-equal vs solo ON the MoM server (prefix caches are model-local —
  the risky upstream token-only-key concern is answered empirically);
  K2 with --no-sticky: trivial follow-up in a heavy conversation re-routed
  and executed by qwen3.5-2b; KV-leak check: fresh trivial conversation
  AFTER heavy traffic still equals the dedicated baseline.
- Root cause of the first vacuous pass: {"thinking":{"enabled":false}} is
  NOT a recognized switch (thinkingSwitch wants bool or {"type":"disabled"}
  or chat_template_kwargs.enable_thinking) — silently dropped, thinking
  stayed on, all tokens went to reasoning_content. Probe bug, not engine.
- Spec gap found + fixed: sticky was always-on; brief requires optional.
  --no-sticky added (commit 2e07609); sticky-on correctly keeps heavy
  conversations heavy.
- Remaining B3: per-lane metrics; then queue priority, B4, B5.

## 2026-11-10 (night, later) — B2 DEVICE PASS (routing proven, commit cf2af3d)

- Single mom-serve process, one endpoint: trivial 'hello' executed by
  qwen3.5-2b; work-verb prompt by qwen3.6-35b; unknown explicit id → HTTP 400;
  sticky follow-up stayed heavy. Observed via reply `model` field = executing
  lane. S1/S2/routing all PASS.
- Tokenizer guard lesson: whole-file tokenizer.json digests too strict — the
  Qwen pair differs only in added_tokens (33 vs 26); vocab.json byte-identical.
  Guard now hashes vocab.json → merges.txt → tokenizer.json.
- Build lesson: host steps don't check unreferenced pub fns of non-root files;
  gate device smokes on the exe build (`if zig build native ...; then smoke; fi`).
- B3 NEXT: trivial-queue priority, reroute KV-isolation device test, per-lane
  metrics in health endpoint; then B4 sleep/wake, B5 classifier slot.

## 2026-11-10 (night) — B1 DEVICE SMOKE PASS (Track B core hypothesis proven)

Window executed per pre-authorization (stop.sh → smoke → start.sh; production
verified healthy after, model Qwen3.8-Flash-Next, warming false).

`zig-out/native/bin/tensorfold-native mom-serve` (kernel-embedded build,
`-Dnvcc=/usr/local/cuda/bin/nvcc`, sm_121):

- S1 N=1: health ok; /v1/models ['qwen3.5-2b']; chat echoed MOM-OK; RSS 429 MiB;
  plan line with reserve "once, not per model". **pass**
- S2 N=2 co-residency: health ok; /v1/models ['qwen3.5-2b','qwen3.6-35b']; BOTH
  engines opened (`qwen3_5 loaded (pinned)` + `qwen3_5_moe loaded`); chat to each
  id answered (B1: both execute on the default engine — B2 routes); single pid,
  RSS 608 MiB host-side. **pass**
- Beside-production refusal path verified earlier: engine budget refused the 35B
  with exact accounting (21.8 GiB vs 19.9 GiB grant = 32 free − 12.2 reserve).
  Plan line prints free=0.0GiB (server module cannot query the device — engine
  budget is the authoritative gate; informational-only plan noted for B2 docs).
- Refusal exit dumps SafeAllocator leak entries from lane-1 internals — cleanup
  nit on the error path only, queued for B2 pass.

B1 acceptance (docs/06): registry+loader co-resident ✓, single endpoint ✓,
advertise all ids ✓ (extra ids ride the --alias path into model_ids), OOM
prevention ✓. NOT yet: per-engine submit (B2 — docs/07-b2-integration.md is the
surgery map; mom_router.zig policy core already green).

## 2026-11-10 (evening) — Track B: M0 + B0 + B1-in-progress

- M0 DONE: Zig 0.17.0 installed (~/.local/bin/zig, exact `.zig-version` match); fork branch
  `mom/main` at upstream `14dea6c6`; pristine baseline worktree at `src/baseline/`.
- B0 DONE (`notes/b0-baseline.md`): pytest 462 pass / 10 skip / 1 known env-timeout;
  `test-golden` 322/335 equal, 0 unexpected; `zig build test` exits 0 but prints 5
  `failed command:` lines — KNOWN-RED list recorded; counts seed-vary (triage rule noted).
- B1 code landed on `mom/main` (all additive; N=1 untouched):
  - `cuda/context_owner.zig` + guarded `Context.deinit` — the first engine to close no
    longer device-syncs or drops a sibling's copy-source spares (2 host tests).
  - `cuda/graph_registry.zig` 8→32 slots + MoM capacity test (drafter caches fit).
  - `server/mom_plan.zig` — pure budget gate: co-residency pass/refuse naming the
    dominating term, quota per model, ONE process reserve (5 host tests, `test-mom-plan`).
  - `server/mom_main.zig` — `tensorfold-native mom-serve --model DIR --name NAME
    [--lane L] [--pin] ...`: resolve/hold all → plan line → sequential `engines.open`
    ×N → serve on default with all ids advertised (3 host tests; B2 routes the ids).
    Refused until their stages: --tp, rank>0, --slide, --recall, --learn-conversations.
  - Dispatch: one line in `server/main.zig`; tests hooked via `server/root.zig`.
- Architecture finding (B2 basis): `api.Engine` vtable (engine_api.zig:425) is the seam —
  MoM becomes a composite engine routing `submit`; `Engine.submit` wrapper validates
  against `e.info()` BEFORE the vtable, so the composite must present a permissive info
  and each child revalidate (verify child strict validation when writing B2).
- Also found: arena/pack alloc hooks are permanent-in-process ONLY for deepseek_v41 —
  co-residing it with a second engine is a documented MoM v1 constraint (hook null at
  steady state for the Qwen pair); context_health refcounts but is SHARED per primary
  ctx → a sticky fault poisons both lanes (process-level semantics for v1).
- B2 policy core landed: `zig/src/server/mom_router.zig` — 1:1 port of the Track A
  Python policy spec (explicit > sticky > trivial-rules > default; work-verb /
  multi-sentence / code bails; case-sensitive id match — the port diverged once and was
  conformed to the spec). 7 decision-table host tests in `test-mom-host` (now 37 total,
  green). Sticky LRU: fixed-cap doubly-linked table, no alloc on decide path, static
  reason strings. Integration hook for openai.zig: caller passes last-user text +
  explicit model field + stable conv key; tools-present requests must bypass trivial.
- Release build: `zig build -Doptimize=ReleaseFast` OK; `zig build native` produces
  zig-out/native/bin/tensorfold-native (dispatches mom-serve). Device smoke next.
- Zig 0.17 API conform pitfalls recorded in the tensorfold-mom skill.

## 2026-11-10 (final) — GATE A PASS; production restored

Full matrix + verdict in `experiments/results/RESULTS.md`. Headline: trivial e2e
0.11s under 6-way big-lane saturation (TTFT p50 4.5ms); big lane e2e matches
dedicated to 0.1%; router p99 75µs; K3 strict 8/8 token-equal; 0/300 trivial leaks,
60/60 real→big after policy tuning (work-verb + multi-sentence hard signals added to
the shared policy spec — char rules alone under-separate, keep for native port).
Experiment servers drained+stopped; production :8000 restored via recipe start.sh.

Next up (Track B / M0+):
- [ ] Install Zig 0.17.0 on spark-2d3b (or provision spark2 access).
- [ ] Create fork worktree `mom/main`; run upstream no-GPU test suite (B0).
- [ ] B1 registry/loader: N × engines.open behind plan-time budget; N=1 parity test.
- [ ] Track A remains deployable: `experiments/` front door with routing.real.json.

## 2026-11-10 (later) — Gate A GPU run in progress

- Derek: production :8000 may be dropped; servers authorized.
- Pulled release v1.2.6 linux-aarch64 (sha256 OK) → `/home/derek/tensorfold-releases/`.
- Checkpoints: `~/models/tensorfold-mom/qwen35-2b` (1.7 G) +
  `qwen36-35b-a3b-nvfp4` (22 G) via hf.
- Production `qwen38-flash-next-tf` container STOPPED via recipe stop.sh
  (restart: `cd /home/derek/recipes/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold && ./start.sh`).
- Lanes live: small :8091 (model 1.57 GiB, 8 streams @ 0.17 GiB @ 8k, loaded 1.3 s,
  greedy default, first-req ttft 15 ms, 168 tok/s); heavy :8092 (model 20.74 GiB,
  5 streams @ 0.90 GiB @ 32k window, 37.2 GiB prompt cache, MTP drafts, loaded 31.3 s,
  thinking default-on). Combined RSS 35 GB vs production's 89 GB.
- Note for Track B planning: each native process keeps its OWN 12.2 GiB reserve —
  two processes pay it twice; a single process would pay once (real MoM win, document).
- Running: reroute_probe → then routing_bench (plain + --saturate 6).

## 2026-11-10 — Phase 0 complete (exploration + docs + Track-A harness)

Done today:
- Cloned upstream `ashhart/TensorFold` @ `14dea6c6` → `src/tensorfold/` (read-only ref).
- 4 parallel subsystem explorations (engine, HTTP server, KV/sessions, Python web) →
  `notes/explore-*.md`; load-bearing claims personally re-verified against source
  (replyModel silent fallback, `Server.engine` singleton @ server.zig:136, submit sites
  chat.zig:249 + warm.zig:58, per-host LaneHost thread, token-keyed prefix trie).
- Docs 00–05 written: brief capture, consolidated findings, per-component options +
  decisions, phased implementation plan (Track A front-door → Track B native fork),
  test plan with v1 qualification gates, model selection (trivial=Qwen3.5-2B-MLX-4bit,
  heavy=nvidia/Qwen3.6-35B-A3B-NVFP4+MTP drafter — both GB10-qualified; brief's
  Qwen2.5-1.5B/Gemma-2-2B rejected: not on the native qualified matrix).
- Track A kit built and *executed*: `experiments/mom_router.py` (shared policy spec,
  15/15 unit tests PASS), `mom_frontdoor.py` (multiplexed children, one endpoint) +
  `smoke_frontdoor.py` 10/10 PASS vs mock engines (routing incl. sticky/explicit/400/
  merged-models/SSE/decision-log), `routing_bench.py` smoke-run (route overhead
  p99 ≈ 20 µs on mocks).

Key decisions (details in docs/02):
1. Two-track: A = multiplexed unmodified native children (fast hypothesis validation,
   stays as safe deployment option); B = native single-process multi-engine fork
   (flagship: N × `native.open`, per-model bundles on `Server`, router at
   `openai.plan()` + submit enforcement). Rejected: cross-model shared batching.
2. KV: per-model pools/stores mandatory (RAM prefix key has no model geometry);
   model id folded into sessions `tag` as defense-in-depth; re-route = re-prefill.
   Plan-time global memory partition; refuse-over-degrade preserved.
3. Sleep/wake via extending `server/idle_unload.zig` (precedent: in-process
   park/reload hot-swap already exists). TP excluded v1 (`cuda/tp_watch` singleton).
4. Exactness inherited per lane (each lane = unmodified pipeline); receipts must run
   per model on the MoM build (tools/exact.py analogue), not just standalone.

Next (M0/M1):
- [ ] Install Zig 0.17.0 (spark-2d3b has none) — or use spark2 (needs host-key fix).
- [ ] Pull the two lane checkpoints; record capacity lines.
- [ ] `routing.real.json`; Gate A run: `routing_bench --saturate` interference numbers
      → `experiments/results/RESULTS.md` + sign/adjust trivial_max_chars.
- [ ] Baseline single-model numbers for heavy lane (sparkDash methodology) for S3.
- [ ] Start B1 (registry/loader) after Gate A passes.

Open questions for Derek:
- OK to run 2 extra `tensorfold-native` servers alongside production :8000 on this
  Spark for Gate A? (Chosen pair adds ~22 GiB weights — should coexist; needs
  confirmation before GPU runs.)
- Is Flash-Next (:8000's model) required as the heavy lane eventually, or is
  Qwen3.6-35B-A3B-NVFP4 acceptable for v1 qualification?

## 2026-11-11 (evening) — S1 fidelity gate complete: item 6 translation MEASURED, NOT VIABLE

Three runs (docs/12 has the full record):
- Run 1 NO-GO was data-thin (2/29 pairs; chat-template mismatch broke token-id
  alignment). Fixed by s1_align.py text alignment (commit 6d082a5).
- Run 2 offline GO (top-1 1.0, cos 0.998) was MEMORISATION: nested-prefix
  checkpoints share every held-out row verbatim with train (519/519).
- Run 3, decision-grade: fit on B5 (6009 rows), evaluate ONLY on a disjoint
  20-turn corpus (5289 novel rows, s1_collect2.py -> results/kvdump-b/):
  best-per-dst mean top-1 **0.3418** (gate 0.99), cosine 0.4071, first-
  divergence 6 tok vs 437/turn, retrieval margins NEGATIVE, controls at
  chance. Ridge maps interpolate one conversation and collapse on new text.

Consequences:
- Linear-translation cache (brief's item-6 wording) CLOSED as measured-not-
  viable; device decode-resume gate never runs (pre-filter failed by miles,
  not margins). docs/10 recommendation updated: translation was the cheap
  road to the 47.3% cold-prefill prize; it is dead.
- The viable item-6 path is the NESTED UPCYCLE: thin student built from the
  35B's frozen layers, fidelity 1.0 by weight identity, shared cache = one
  physical copy via Paged page-table views; GDN heads via duplication;
  student-only distillation (incl. reading heavy-lane thinking blocks in
  history; Derek's question answered: template unification is a serving
  config fix for measurement, thinking-behaviour is free by inheritance for
  the nested model).
- Harness durable: TF_MOM_KVDUMP dump hook (059a51e incl SafeAllocator
  regression test), s1_align.py, s1_fit.py --eval-corpus + verbatim-leakage
  stat (never trust same-conversation holdout again).

## 2026-11-11 (night) — Nested upcycle: spec (docs/13) + M0 extraction tool

- docs/13-nested-upcycle-spec.md: frozen-prefix construction (correction:
  scattered-layer extraction cannot share rows — row identity needs residual-
  stream identity, so thin = heavy.layers[0..N] bit-identical ++ trainable
  tail; GDN states shared too, better than the attention-only sketch).
  Grounded on real tensor inventory (layer ~465 MiB, 93% experts; prefix
  0..9 = 4657 MiB; embed 970 MiB BF16; QK-norm inside projections) and the
  real cache code (Paged = base+ids indirection over core pool; Workspace
  leases are BINARY leased:bool -> v2 needs refcount; reuse kept-state is the
  v1 adoption path — copy-based, no kernel surgery).
- GDN 16v->32v: NOT a prefix trick (prefix keeps heavy 32v bytes; states are
  the heavy engine's own); legit ONLY as tail-side weight-tying by
  construction (train 16v, export pair-duplicated 32v, structural assert).
- Distillation: teacher = heavy engine's own logits (deployment NVFP4
  numerics); data mix incl. thinking-blocks-in-history (lane hops) and
  heavy-prefilled prefixes; gate G1 = kvdump bit-identity of layers 0..9
  across engines (the exact-decode guarantee as regression test).
- M0 tool live: experiments/nested_extract.py (pure stdlib safetensors byte
  copy, never requantizes; warm-start tail = parent's next 6 layers so the
  student starts as a coherent 16-layer truncation — removes M1 loader shape
  risk; emits NESTED.json frozen_prefix_digest = adoption permission;
  --verify byte-hashes the frozen set vs parent).

## 2026-11-11 late: M1 + M2 PASS (nested program)
- M1: engine opens nested-16l. Root cause of MissingTensor layers.16 was
  DEAD CONFIG (engine reads text_config; top-level num_hidden_layers ignored).
  gibberish reply = expected pre-distillation floor (G2's gate, not M1's).
- M2 teacher-forced PASS: 9 ckpts, tokens equal, K+V layers 3/7 BIT-IDENTICAL
  across engines (results/m2-tf/kvdump, m2_tf_score.py).
- Tooling committed 5c5e301 (extractor, M1/M2 probes, gdn_dup selftested).
- Distillation scripts written (distill_collect/ train); HF supports
  qwen3_5_moe_text causal LM -> trainable student from nested ckpt plausible.
  NEXT: M3 adoption hook (cuda_reuse kept-state strided DtoD), then collect+
  train window; production left DOWN per Derek.

## 2026-11-12: M3 CLOSED + materializer/distill + M4 adopt-time wiring
- M3 gate PASS (G3 window, tail-zero policy c179d63): adopting the parent's kept
  state copies regions only inside the frozen prefix (frontier from
  NESTED.json frozen_layers); tail GDN regions cold-start — tail layers are
  head-duplicated by design, parent bytes are foreign there. Probe rounds 0-4:
  adoption fired, decode finite, replies byte-identical across boots.
- Materializer bug root-caused + fixed (03928f5): some non-expert NVFP4 tensors
  are fp8_e4m3 SCALAR-quantized — must multiply {name}_scale before the bf16
  cast (v_proj std was 36.9 vs true 0.0126, ~3e3 inflation).
- Distillation pipeline proven end-to-end: collect (175 rows), train 2 epochs
  (loss 4.70 -> 1.14), per-layer streamed safetensors save (6fe21d0) sidesteps
  the save_pretrained cgroup OOM. Checkpoint
  ~/models/tensorfold-mom/qwen36-nested-16l-t1 still gibberish = 175-row smoke
  corpus too small, NOT a pipeline fault; scale the corpus for G2.
- HF is NOT a usable GDN third reference (naive torch_recurrent AND fla chunk
  paths corr~0 vs the bit-identical engine pair) — arbiter retired; logprobs
  unsupported on this backend (greedy continuation is the logits-gate proxy).
- M4 producer wiring (9b1ebc4): TF_MOM_SHARED_VIEW=1, both engines bf16+paged:
  parent keep pins its whole KV pages (pool retain, recorded in AdoptSink
  .shared); thin adopt SKIPS their copy, injects the parent's physical page ids
  into device-table slots [0..n), sets cache.shared so Cache.seg resolves
  skv/sps/shp into the parent arena (kernel resolution from cbfb3ec). Writes
  never leave the thin cache's own pages; partial boundary page + state block
  stay on the copy path; every gate miss falls back to copy. Pages.close
  releases the thin side's own pins. Probe round 5 gates: resolve log line
  present (no silent fallback) + zero-copy reply byte-identical to copy path.
  NEXT: native build (zig build native -Doptimize=ReleaseFast -Dnvcc=..., from
  repo ROOT) + device window: stop prod :8000, run_g3.sh under
  systemd-run --user --scope -p MemoryMax=100G, verify M4 gate, restore prod.

2026-11-12 (afternoon) -- third-party survey: mergekit & kessel.
  mergekit (arcee-ai, LGPL-3.0, active): YAML weight-space surgery; refuses/
  dequantizes FP8/FP4 block-quantized models -- cannot touch our NVFP4 parent;
  zero GDN support. Harvested design ideas only (declarative slices, refuse-
  by-default, multi-stage pipeline, deterministic per-tensor RNG).
  kessel (Schneewolf-Labs, MIT, (c) 2026 Nick Beerbower): mergekit-parity
  engine that handles block-quantized FP8/FP4 (DeepSeek-shaped scales).
  Ideas adopted-with-credit (no code vendored): overlay/delta checkpoints
  (symlink base shards + delta index) for distillation outputs; requant
  audit counters (blocks_rescaled/bytes_changed) as positive controls;
  promote-changed -> bf16 + ignored_layers/modules_to_not_convert dual-list;
  applied-config written into artifact (kessel_config.yaml pattern); config
  remap recursing into *_config (validates our M1 text_config fix) with
  dtype-vs-torch_dtype key preference. Their quant.py assumes 2-D block
  scales; our ModelOpt-style scalar weight_scale is unsupported -- nested
  byte surgery stays in nested_extract.py. Decision: stay separate; optional
  future upstream = generic ModelOpt NVFP4 codec PR to their quant.py.
  Design inspirations credited here and in the headers of whatever we build
  from them.
