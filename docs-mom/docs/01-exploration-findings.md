# Exploration findings (consolidated)

Deep subsystem maps live in `notes/explore-{engine,server,sessions,web}.md`
(produced 2026-11-10 against upstream HEAD `14dea6c6`; load-bearing claims personally
re-verified against source). This is the decision-relevant digest.

## 1. The engine is already instance-based — "one model per process" is a *server* policy

- `core/engine_api.zig:421 Engine` is a ctx+vtable trait object; `native/cuda.zig:51 Host`
  ("one loaded model behind the lane host") is heap-allocated per open, with its own
  family state, buffers, KV ledger, prompt store, and **its own lane thread**
  (`core/lane_host.zig:126` spawns 16 MiB thread per host).
- The single-model property comes from `server/main.zig`: exactly one `engines.open`
  (`:154`), one `serve.run` → `Server.init(gpa, io, s.engine, ...)` (`serve.zig:229`).
- **The process already holds 2–4 neural networks per serve**: target + drafter/MTP
  (`native/cuda_drafted.zig`, a full second net with own weights/kernels), vision tower
  (`families/nemotron/cuda_engine.zig:196`), and the Sliding-Weights trainer
  (`native/cuda.zig:412`). Second-net co-residency machinery *exists and works* —
  it is just always "one *target*".
- Dynamic in-process open/close already happens: `--idle-unload` parks and reloads
  engines at runtime (`server/idle_unload.zig:195,274` swaps `srv.engine`/`srv.info`
  under a mutex). This is the precedent for sleep/wake and the hot-swap precedent for
  per-model engine tables.
- Verdict from the engine map: **two `Host`s in one process is highly feasible**; the
  work concentrates in `server/`, plus a handful of CUDA-layer globals
  (`cuda/memory.zig` single allocation limit; `cuda/context.zig:57` deinit drops
  *shared* copy-source spares; `cuda/graph_registry.zig:23` 8 report slots;
  `cuda/copy_source_budget.zig` shared budget; `cuda/tp_watch.zig` TP singleton →
  exclude TP from v1).

## 2. Request path: one clean router seam, and a silent-fallback quirk

Path: accept thread (`serve.zig:291`) → per-connection thread (`server.zig:431`) →
`routes.dispatch` (`routes.zig:35`, admission/idle bookkeeping) → `openai.post` →
**`openai.plan()` (`openai.zig:129`) is the only place the parsed `model` field is read.**

- **Today the `model` field is never validated for chat/completions/responses/messages**:
  `replyModel` (`openai.zig:110`) falls back to `served_name` silently — ask for
  "gpt-99" and the served model answers, replying with your asked name if it happens to
  match an alias. (Contrast: `/v1/embeddings`, `/v1/audio/speech`, `/v1/images` DO 400
  with "model {s} is not served here".) `api.Request` has **no model field** at all —
  the engine never sees a model name.
- Text-generation submit happens at exactly two sites: `server/chat.zig:249`
  (`srv.engine.submit(...)`) and `server/warm.zig:58` (background prefill pass). Both
  read `srv.engine` — the single `api.Engine` field on `Server` (`server.zig:136`).
- Scheduler: there is **no global scheduler singleton** — the continuous-batching loop
  lives inside each engine's `LaneHost` (`core/lane_host.zig:35`, queues `queued` /
  `admitted` / `filling`, `run` loop at `:504`; chunked prefill via
  `lane_host_fill.zig`, decode rounds over live streams via `lanes.Engine.step`).
  Per-model queues therefore come *for free* with per-model engines.
- `/v1/models` (`status_routes.zig:77`) emits one entry per `config.model_ids`, all
  carrying the single engine's `srv.info`.

## 3. KV / sessions: no conversation ids; token-prefix tries; model identity only on disk

- Identity is **(tag: u32, token-id prefix)** — `sessions/prefix.zig` chains
  Blake2b-128 digests of token pages under `"tf-sess-root"||tag`; entries keyed
  `"tf-sess-entry"||node||pos||tail`. There is **no conversation id** anywhere;
  "a chat" = a leaf entry's lineage (`isLeaf`/`members`).
- `tag` is family-interpreted mode/shape namespace (dsv41: replay vs fast), **not model
  id**. The RAM-side key contains nothing about checkpoint, layers, or dims → two models
  sharing one `Store` would collide on identical prefixes and `adopt` the *wrong model's
  pages* silently (no geometry guard on the RAM path; the disk path checks family names
  and a `compat` SHA256 digest per file).
- Consequences (confirmed): **one Pool + DevicePool + Store per model is mandatory**
  (`Pool.init` fixes layout forever; device tensors pre-allocated). The `tag` u32 is the
  natural namespace hook (already flows through `root`/`entryKey`/`Head.tag`) but is not
  sufficient alone; keep per-model Store instances AND fold model id into `tag`.
- Re-routing a conversation to another model = the new model re-prefills from its own
  prefix cache (correct semantics), never KV transfer. Cross-model KV translation is
  genuinely a new format (per-family row bytes, compat-gated) — phase 2, matching the
  brief.
- Tiers: HBM page pool (per family tensors) / host RAM bounded state / NVMe
  (`sessions/disk.zig`, `O_DIRECT`, per-chunk SHA-256, LRU trim, per-compat dir +
  flock). All instance state — two models get two disk trees for free.
- Budgeting is process-global: `native/cuda_memory.zig` computes `room = free − reserve`
  once (unified default reserve = total/10, floor 4 GiB; the "9 GiB free" in the recipe
  README is this formula's output, not a constant); `cache_fit.footprint()` is a
  process-wide high-water. **MoM must partition the device budget at plan time** (sum
  both models' weights+pools against one reserve; refuse over-budget at startup —
  preserve TensorFold's refuse-over-degrade contract).

## 4. Python web layer: no mux today, but every brick exists

- `cli.py` holds a per-user lock (one Web per user), builds **one** `Engine`
  (`engine.py:226` refuses a second: "an engine is already running"), one relay
  (`server_relay.py` → `proxy.py:Proxy`). The `model` field is never read — the relay
  forwards opaque bytes to the single `engine_url`.
- But `Engine(settings, state_dir)` is already instance-parameterized; `Proxy`'s
  `target` callable is per-request by design ("a change of engine midway never sends
  one's key to the other"); `fleet.py` (16-peer registry), `engines.identify()`
  (per-server `/v1/models` probe), `conversations.py` settings-merge (designed
  extension point), `usage.py` per-(hour,model) stats, and `series.py` scripted
  stop→start→wait all exist to reuse.
- No routing policy layer at all; no auto-restart; memory budgeting assumes one
  resident engine (`native.py:336`, `fit.NATIVE_COPIES`).

## 5. Surprise list (things the brief assumed differently)

1. There is no "the KV Cache Manager" component to fork per se — the generic native
   server uses `core/kv_pool.zig`/`core/prompt_cache.zig`; the sessions
   prefix/disk stack is owned by the dsv41 family. MoM must handle **both** shapes.
2. "Shared continuous-batching loop" (brief §3) conflicts with reality: batching is
   engine-internal (`LaneHost`). Cross-model shared batching would mean touching exact
   round scheduling. Decision recorded in `02-component-options.md`: keep loops
   model-local, deliver trivial-priority via admission control.
3. The engine silently serves any unknown `model` string — MoM turns a quirk into the
   routing hook, but must restore explicit 400s for truly unknown ids to keep client
   expectations honest.
4. Sleep/wake already half-exists (`idle_unload` close/reopen in-process, rank
   followers, retained-Gpu registries) — we are extending a live mechanism, not
   inventing one.
5. Weight paging/streaming from NVMe for MoE experts is already shipped (FSR,
   `docs/design/fluid-subgraph-residency.md`) — the brief's registry requirement is met
   *for MoE weights* by upstream; MoM adds per-model, not per-expert, residency states.
