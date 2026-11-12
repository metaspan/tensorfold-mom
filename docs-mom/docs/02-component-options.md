# Component options and decisions

Each section takes one item from the brief (`00-design-brief.md`), lists the real
options given the explored codebase, and records a decision with rationale.
Citations are `file:line` in `src/tensorfold/` (upstream HEAD `14dea6c6`).

## 0. Fork realization (the meta-decision)

| Option | What | Pros | Cons |
|---|---|---|---|
| **A. Native Zig multi-engine** | N `api.Opened` from `native/cuda.zig:155 open` in one process; `Server` holds a per-model table; router at `openai.plan()`/submit | The brief's literal goal; zero extra IPC hops; shared CUDA primary context already refcounted (`cuda/context.zig:23`); shares one accept loop, one `/v1/models` | Touches `server/` core (hot path); needs CUDA global accounting fixes (memory limit, context spares, graph registry, copy budgets) |
| **B. Multiplexed children (front door)** | One Python process (extend `web/tensorfold_web/` pattern) running N stock `tensorfold-native serve` children, one public endpoint, policy router | Days not weeks; zero exactness risk (unmodified engines); validates routing policy + traffic separation + memory co-residency now; every brick exists (`Engine` instance-parameterized, `Proxy.target` seam, `fleet.py` registry, `engines.identify`) | Not literally single-process (N engine processes); per-request hop to localhost child; global memory budget is advisory across children |
| C. Full engine merge (one LaneHost batching across models) | Shared continuous-batching loop mixing streams from two models in one round | Brief §3's "shared continuous-batching loop" taken literally | Breaks per-engine round scheduling and exactness contracts; no precedent; massive risk |

**Decision: A is the flagship; B runs first as the hypothesis test (Phase A) and stays
as the deployment-safe fallback.** C is rejected for v1: the brief itself keeps
prefill/in-flight batching model-local, and each `LaneHost` round is a model-specific
kernel pipeline (`lanes.Engine.step`, `lane_rounds.zig`) whose exactness was qualified
per family. Cross-model co-scheduling buys nothing measurable until one model is
memory-bound (then: priority admission, see §3).

## 1. Model Registry & Weight Manager

Options:
- (a) Load all models eagerly at startup, resident forever. Simple; costs weights RAM
  even when idle. Our chosen pair (~22 GiB weights total) affords this.
- (b) Extend `idle_unload` (`server/idle_unload.zig`) per model: park (close engine,
  keep config) / wake (reopen; load takes model-load time) — already works in-process
  for the single model (`:195` stub swap, `:274` reload). Gives "sleep the large model"
  cheaply: park = close + retain identity; wake = reopen with warm page cache
  (`core/checkpoint_cache.zig` fadvise hygiene, `digest_cache` warm digests).
- (c) True weight paging/streaming of whole-model weights from NVMe — already solved
  *within* a model by FSR for MoE experts (`docs/design/fluid-subgraph-residency.md`);
  inventing a second mechanism at model granularity is unnecessary for v1: (b) + FSR
  covers "pin the small, sleep the big".

**Decision:** N eager-opens behind a registry (model id → `Opened` + info + text +
policy), with per-model park/wake states reusing `idle_unload` machinery, and a global
memory plan computed *before* any open (sum of `native/cuda_family_memory.zig:46
weights` estimates + pools + one reserve; refuse startup over budget — preserve
refuse-over-degrade). Pinning = "never park" flag per model (small model default
pinned).

## 2. Request Router

Hook-point options (from the server map):
- (a) `routes.post()` (`routes.zig:114`) — earliest, but body unparsed; would re-scan
  raw bytes.
- (b) `openai.plan()` (`openai.zig:129`) — the only parsed-model site; one hook covers
  chat, completions, responses, AND anthropic messages (they all funnel through
  `runNamed`→`plan`). Today's silent fallback (`replyModel`, `openai.zig:110`) lives
  here: replace fallback semantics with routing.
- (c) `chat.generate()` (`chat.zig:220–256`) just before `srv.engine.submit` (`:249`) —
  carry the chosen engine on `chat.Prepared`; second submit site `warm.zig:58` must use
  the same choice.

**Decision:** decide at (b), enforce at (c). Router is a pure function
`decide(model_field, body, headers, load_table) -> {model_id, reason}` running on the
connection thread before any GPU work: string compares + a token-count heuristic on
prompt char length (tokenizer runs later in `chat.prepare`; char/word proxies suffice
for the <5 ms budget and are all the v1 rules need: length < N, trivial patterns,
code fence or language keywords present, `tools`/images present → big, explicit model
field wins, sticky `x-tensorfold-conversation` (or first-message hash) LRU overrides
rules when enabled).
Router config: a small `routing.json` (loaded at start, hot-reloadable), NOT a
settings-kernel change. Unknown `model` ids become explicit 400s (matching
embeddings/speech behavior, `embeddings.zig:18`) rather than today's silent fallback —
except ids that match a *registered* MoM model, which route.
Optional tiny classifier and load-aware routing are v2 slots behind the same interface
(load table is already available per engine: `/health` `live.connections`/`live.waiting`).
**Derek 2026-11-10: classifier-based routing is explicitly a later enhancement** — v1
ships rules-only (with the hard-signal set Gate A proved mandatory); the Decision
interface and routing.json must keep a pluggable `policy: classifier` slot.

## 3. Scheduler & Batcher

Brief asks: "per-model queues with a shared continuous-batching loop; trivial-model
queue has higher priority".
Reality: queues and the batching loop live inside each engine's `LaneHost`
(`core/lane_host.zig:35`; `queued`/`admitted`/`filling`, run loop `:504`) — per-model
queues come free with per-model engines; a *shared* loop is option C (rejected).

Trivial-queue priority options:
- (a) Server-level admission control: the router hands requests to their engine's
  `submit`; the small model's engine always has spare capacity (its streams are
  separate), so trivial requests *already* never queue behind big-model work except on
  GPU timeslicing. Residual interference = CUDA SM contention while the big engine
  runs a round.
- (b) Explicit priority: cap big-engine concurrency (`--parallel`) and give the small
  engine unconditional admission; optionally a `cudaStreamCreateWithPriority`
  low-priority stream for the big engine's lane (CUDA supports stream priorities;
  needs a probe that lanes honor it).

**Decision:** (a) for v1 (measure first — the brief's own success criterion is
"latency indistinguishable from dedicated"; interference gets measured in Phase A/B,
not assumed). (b)-stream-priority is the pre-approved fallback if measurement shows
interference > threshold.

## 4. KV Cache Manager

Options:
- (a) Independent paged caches per model — **mandated**, not chosen: the RAM prefix key
  is `(tag, token pages, pos, tail)` with zero model geometry in it
  (`sessions/prefix.zig:21,37`), and `Pool.init` fixes layout/device tensors forever
  (`sessions/pool.zig`). Two models sharing a Store silently cross-adopt wrong pages.
- (b) Namespace one shared store by model id — rejected: pool tensors are per-layout,
  can't coexist in one Pool; the tag fold is needed *in addition* anyway.

**Decision:** each engine already owns its pools (generic: `core/kv_pool.zig` +
`core/prompt_cache.zig`; dsv41: sessions Stack) — MoM keeps that and adds: (i) model id
folded into sessions `tag` (defense-in-depth), (ii) `restore.Want`-style identity
verification on RAM adopt (the disk path already checks compat; port the guard to the
generic prompt-cache path if absent), (iii) a global KV budget computed at plan time
(§1) splitting `native/cuda_memory.zig` `room` across engines — each engine then sizes
its streams from *its quota*, not process free memory. Prefix caching stays per model;
re-routed turns re-prefill (correct, fast on the small model). Cross-model KV
translation (PrefillShare/linear-map) stays phase 2 behind the same extension point:
compat digest + format.zig already gate any cross-model file reuse.

## 5. Execution Pipelines

No change (per the brief): each `Host` keeps its family kernels, lane engine, drafter
and exact verification. The drafter path (`native/cuda_drafted.zig`, second net with
own kernels) is proof the pattern composes. MoM adds **no** new kernels. One rule:
exactness receipts (`tools/exact.py`-style drafted-vs-plain, concurrent-vs-solo) must
run per model on the MoM build, not just per model standalone.

## 6. Memory & Runtime Orchestrator

Required changes (from the engine map), all in `zig/src/cuda/` + `native/`:
- `cuda/memory.zig:11` single device accounting + `setAllocationLimit`: add per-owner
  quota accounting (keep one physical counter; add per-engine ledger + soft caps so
  engine A cannot starve engine B's KV admission).
- `cuda/context.zig:57` `deinit` device-syncs and `copy_source.dropSpares` for the
  shared primary context: refcount per owner so parking engine A doesn't drop B's
  spare copies; skip device-sync when other owners live (schedule-sync per stream).
- `cuda/graph_registry.zig:23` 8 fixed report slots → grow or per-engine slab.
- `cuda/copy_source_budget.zig` shared capture budget + 64 scopes: give each engine a
  sub-budget; concurrent capture is untested → host test first.
- `native/cuda_family_memory.zig:46` estimate per open → plan-time sum before opens.
- Sleep/wake: reuse `idle_unload` park/reload with per-model policy (pinned small
  model; big model parks after `idle_after` s, wakes on request with measured latency).
- TP excluded from v1 (`cuda/tp_watch.zig` process singleton); MoM refuses to start
  with a TP-configured model. Multi-node TP/PP "remains per-model" per the brief —
  honored by not-break: a MoM process runs only non-TP models.

## 7. Observability

Additions to `server/metrics.zig` + `/health`:
- `routing_decisions_total{model,reason}` + per-decision route latency histogram
  (validates the <5 ms budget from inside the server, not just from the client).
- Per-model: reuse existing per-request `first_ns` TTFT (`openai.zig:50,96,291`),
  streams, queue depth (`/health live.connections`/`live.waiting` already per engine →
  expose per model in the merged view), cache hit/miss from `prompt_cache`/`Store`
  counters, decode tok/s.
- Memory: per-engine quota usage vs cap; plan-time verdict line.
- Switch latency: park/wake timestamps from the idle manager.
- Dashboard hooks: the existing `/metrics` Prometheus shape extended with a `model`
  label (the TensorFold Web `usage.py`/`history.py` already consume per-(url,model) —
  the front-door variant feeds it unchanged).

## Routing Policy (v1) — settled shape

```
1. body.model names a registered model            → that model   (reason=explicit)
2. sticky: conv key in sticky-LRU (if enabled)    → sticky model (reason=sticky)
3. trivial rules (any hit)                        → small model  (reason=trivial:<rule>)
     - prompt char length < N (default 240 ≈ 32-64 tok)
     - matches trivial pattern list (hello/status/ack/thanks/…)
     - no `tools`, no image/audio parts, single user turn
4. otherwise                                      → default model (reason=default)
```
Free re-routing is the default (rule 3 runs every message); stickiness is opt-in
per config and applies per incremental message until conversation end.
Every decision is logged in the response `usage`/headers for observability
(`x-tf-routed-model`, `x-tf-route-reason`, `x-tf-route-us`).
