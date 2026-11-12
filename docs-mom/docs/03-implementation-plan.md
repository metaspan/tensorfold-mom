# Implementation plan

Two tracks, per `02-component-options.md` §0: **Track A** (hypothesis validation on the
multiplexed-children front door, days) feeds **Track B** (the native single-process
fork, the flagship). The brief's own implementation order maps onto Track B phases B1–B5.

Environment note: builds need Zig 0.17.0 (`ZIG-PREVIEW.md`); spark-2d3b has none yet —
install is prerequisite W0.1. The fork branches from upstream `main` at the pin below;
rebase routinely (upstream is very active — ~10 commits/day at clone time). Upstream PRs
require ATC claims; fork-internal work does not.

Pin: `ashhart/TensorFold` @ `14dea6c6` (2026-11-10). Track B branches `mom/main`.

---

## Phase A — Hypothesis validation (multiplexed children; no Zig changes)

Goal: prove the traffic-separation hypothesis and harden the router policy *before*
touching the engine. Deliverable: `experiments/` results answering the brief's minimal
experiment, plus a production-usable fallback architecture.

- **A1. Front door** (`experiments/mom_frontdoor.py`, standalone, stdlib+http only):
  proxy one public port → 2 local `tensorfold-native serve` children
  (Qwen3.5-2B :8091, Qwen3.6-35B-A3B-NVFP4 :8092 per `05-model-selection.md`).
  Implements the §`02` routing policy verbatim (shared with Track B as the *policy
  spec*), merged `/v1/models`, per-child health, sticky LRU, decision headers, JSONL
  decision log. (Informed by `notes/explore-web.md`; if the Web layer is preferred as
  host later, port `router.py` in unchanged — `Proxy.target` is the seam.)
- **A2. Routing harness** (`experiments/routing_bench.py`): mixed synthetic workload
  (trivial set: greetings/acks/status/short QA; real set: code, long QA, tool calls),
  controlled concurrency; measures per-route TTFT/e2e, router overhead (front-door
  in-server timestamp vs client), interference (trivial TTFT with big engine idle vs
  saturated), memory headroom (`free`, engine capacity lines), park/wake proxy
  (kill/respawn child time = reactivation floor).
- **A3. Re-route correctness probe** (`experiments/reroute_probe.py`): same conversation
  alternated across models per turn (forced `model` field), asserted against solo-run
  ground truth at temperature 0.
- **Gate A → B:** trivial-route TTFT within noise of dedicated small server (≤ +10%),
  trivial traffic share matches rule expectations, no KV cross-talk observed, memory
  plan fits. If the hypothesis fails (e.g. interference dominates), stop and re-plan
  before writing Zig.

## Phase B — Native fork (the deliverable)

Order follows the brief's implementation order.

- **B0. Harness-first:** fork repo + CI-shaped script running upstream's no-GPU test
  suite (`zig build test` subset per `docs/evaluating.md`) so every later step lands
  green. Plus: reproduce the single-model baseline numbers (decode/prefill/TTFT via the
  recipe's sparkDash methodology) to compare "normal TensorFold performance preserved".
- **B1. Multi-model registry & loader** (brief order 1):
  - `server/main.zig`: accept N positionals / repeated `--model DIR --name ID [--lane
    small|big] [--pin] [--idle-after S]`; plan-time budget sum (`native/cuda_family_memory.zig`
    estimates + pools + one reserve) *before* opens; N × `engines.open`; registry table
    `{id → Opened, info, text, served_name, aliases, policy}` on `Server`
    (`server.zig:133` gets `models: []ModelBundle`, keep `engine`/`info` as the default
    bundle so single-model behavior is byte-identical when N=1).
  - CUDA globals: per-owner context refcount (`cuda/context.zig:57` spares drop fix),
    per-engine quota ledger (`cuda/memory.zig`), graph-registry slots ≥ 8×N,
    copy-source sub-budgets. Host tests first (concurrent capture).
  - Refuse: TP models, >1 GPU (v1 single device; `api.Open.device` keeps the door open).
  - **Exit:** `serve --model A --model B` loads both, `/v1/models` lists both ids with
    real per-model `info`; memory line prints the plan; N=1 path unchanged (exactness
    receipts pass identically).
- **B2. Request router** (brief order 2): `server/router.zig` (new, pure, host-tested):
  policy engine from `02-component-options.md` §7, fed `routing.json`. Hook at
  `openai.plan()` (replace `replyModel` fallback; explicit 400 for unknown ids) and
  enforce at `chat.generate`/`warm.zig` submit via `Prepared.model`. Decision metadata
  into response headers + metrics counters. <5 ms asserted by host microbench
  (target: <100 µs for rule path).
- **B3. Scheduler dispatch** (brief order 3): per-model `LaneHost` already exists —
  work is (i) admission/idle bookkeeping keyed per model
  (`srv.load.enter(group)`, `idle_manager`, `admission` lease in `routes.zig:114`
  reviewed per-model), (ii) big-lane concurrency cap option, (iii) A2 interference
  measurement replayed on the native build; stream-priority fallback if needed.
- **B4. Independent KV + correct prefix caching** (brief order 4): model id folded into
  sessions `tag` (and generic prompt-cache `sharedKey`); RAM-adopt identity guard on the
  generic prompt cache (`restore.Want` analogue); per-engine KV quota from the B1 plan
  (each engine sizes streams from quota, capacity lines reflect it); per-model disk
  namespaces already separate (compat dirs) — verify flock paths differ.
- **B5. Observability & sleep/wake** (brief order 5): metrics per `02` §7; extend
  `server/idle_unload.zig` to per-model states (pinned / idle-parked / parked),
  park-on-idle for unpinned lanes, wake-before-route on parked targets, switch-latency
  histogram; `/health` merged per-model view.
- **B6 (phase 2, gated):** shared prefill / cross-model KV translation experiments —
  not scheduled here; extension points reserved (compat digest, format.zig, PrefillShare
  notes in `01` §3).

## Phase C — Prove v1 acceptance

Run the brief's Success Criteria as a formal qualification on the native build (see
`04-test-plan.md` §V1), on spark-2d3b with the chosen pair, production :8000 untouched
(maintenance window only if Flash-Next must be the heavy lane).

## Milestones & rough effort

| M | Contents | Gate | Est. |
|---|---|---|---|
| M0 | W0.1 zig install; clone pinned; test suite green; baselines | env ready | 0.5 d |
| M1 | Phase A complete, report in `experiments/RESULTS.md` | Gate A | 2–3 d |
| M2 | B0+B1 | two models co-resident, N=1 unchanged | 3–5 d |
| M3 | B2+B3 | single endpoint routes; router overhead proven | 2–4 d |
| M4 | B4 | reroute probe passes natively | 2–3 d |
| M5 | B5 | sleep/wake + dashboard metrics live | 2–3 d |
| M6 | Phase C | v1 success criteria signed | 1–2 d |

## Risks & mitigations

1. **CUDA global accounting bugs under 2 engines** (context spares, capture budget):
   host tests before device; land each fix independently upstream-shaped.
2. **Upstream churn / merge pain:** keep fork deltas in new files + ≤ ~30-line hooks in
   hot files; rebase weekly; prefer upstream accepting small PRs (ATC-claimed).
3. **Interference >10% on trivial lane:** stream priorities; else last-resort
   MPS/green-contexts experiment; else Track A deployment (children) remains viable.
4. **Qwen3.5-2B CUDA path maturity** (quick-start qualified, but the 2B recipe was
   authored Metal-first): run its receipts first on this Spark before building on it;
   fallback trivial model: `LiquidAI/LFM2.5-24B-A2B`? No — too big. Real fallback:
   a second `qwen3_5`-family small ckpt or the `llama` family's smallest qualified
   checkpoint.
5. **dsv41 vs generic cache duality:** the chosen pair uses the generic path; sessions
   stack changes are exercised only when a dsv41 lane joins — keep them behind the same
   tag-fold so the semantics are uniform.

## Out of scope (v1)

TP/PP MoM mixes; multi-GPU MoM; shared/cross-model batching; cross-model KV transfer;
classifier-based routing; joint training of any kind; Windows/HIP/XPU/Metal backends
(CUDA linux-aarch64 only for qualification; code stays backend-neutral where free).
