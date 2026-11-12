# TensorFold Zig HTTP server — request lifecycle, model field, scheduler (for the MoM fork)

Explored read-only at `/home/derek/tensorfold-mom/src/tensorfold/zig/src/` (server/ + core/).
Note: there is no `http.zig`; the HTTP layer is `server/server.zig` (lifecycle/dispatch) + `server/http_conn.zig` (framing) + `server/routes.zig` (routing).

## 1. Request lifecycle: socket accept → first token

### Startup (one engine, one endpoint)
- `server/main.zig:154` — `startup.both(...)` loads tokenizer text and engine **in parallel**: `engines.open(a, gpa, io, dir, model_type, args, &problem)` (`native_engines` module; e.g. `native/metal.zig:127 pub fn open` or `native/cuda.zig:155`). It opens **exactly one** engine, returns `api.Opened{engine, close, halt}`.
- `server/main.zig:166` — `const served = hub.servedName(args.name, args.model, dir)` (`server/hub.zig:65`: `--name` if given, else basename of the model dir/repo id).
- `server/serve.zig:134 run(gpa, io, args, Setup)` — builds `Config`:
  - `serve.zig:149–156`: `ids` = `[served] ++ args.alias`, deduped → `config.model_ids` ("The served name first, then aliases" — `server.zig:49–52`).
  - `serve.zig:181–228` — the `server_mod.Config` literal (`.served_name = s.served`, `.model_ids = ids.items`, `.background_ids` from `--name-priority`).
  - `serve.zig:291` — accept loop runs on **one dedicated thread**: `std.Thread.spawn(.{}, server_mod.Server.serve, .{ srv, lis, &accept_stop })`.

### Per connection
- `server/server.zig:426 Server.serve` — `listener.accept(stop)` (`server/listener.zig:85`), `srv.admitConnection(fd)` (`http_admission.zig` max-connections), then **one OS thread per connection**: `std.Thread.spawn(.{ .stack_size = 8 << 20 }, connection, .{ srv, c, loopback_bind })` + `t.detach()`.
- `server/server.zig:447 connection` — per-connection arena, keep-alive loop:
  `conn.readRequest(a, !first)` (framing in `server/http_conn.zig`) → `request_guard.allow` (browser/CORS guard) → `srv.authenticate` (key + `auth_scope.allows`, `server/auth_scope.zig:71`: scopes `inference|read|admin`, exact route match) → `srv.dispatch(&conn, a)` → `routes.dispatch`.

### Routing
- `server/routes.zig:35 dispatch` → `post()` (`routes.zig:114`). Before any handler, `post()`:
  `srv.generating`++ → `srv.load.enter(group)` (`load_ledger.zig`, group `.chat_completions`/`.responses`/…) → `srv.idle_manager.begin()` → drain check (503) → `srv.admission.acquire` (`request_admission.zig` lease, per-key in-flight/RPS) → capability gates (embeddings/rerank/image/readout engines refuse wrong routes) → `idle_unload.wakeReload(m, srv, a)` (reloads a parked engine).
- Chat/completions fall through to `openai.post(srv, conn, a, is_chat)` (`routes.zig:199`).

### JSON parse and the `model` field
- `server/openai.zig:45 post` — `http_body.readJson(conn, &cx)` is where **the OpenAI request body is JSON-parsed**; then `run`/`runNamed` (`openai.zig:236/241`).
- `server/openai.zig:129 plan` — all validation happens here. The `model` field is handled in two places only:
  1. **Copied verbatim into the sampling fields** (`openai.zig:157`):
     ```zig
     for ([_][]const u8{ "temperature", ..., "seed", "priority", "model", "draft", ... } ++ grammar.fields) |k|
         if (body.get(k)) |v| try sampling.put(cx.a, k, v);
     ```
     so it rides into `chat.Input.fields` and is later re-read for `--name-priority` only.
  2. **`openai.zig:110 replyModel`** — name resolution for the *reply*:
     ```zig
     pub fn replyModel(srv: *const Server, body: Value) []const u8 {
         if (body.get("model")) |m| if (m == .string) for (srv.config.model_ids) |known|
             if (std.mem.eql(u8, known, m.string)) return m.string;
         return srv.config.served_name;
     }
     ```
     used as `Plan.named` (`openai.zig:188`) and echoed in every chunk/body (`openai.zig:284,400`), and by `responses.zig:187` and `anthropic.zig:85`.
- **There is NO validation of `model` for chat/completions/responses/messages today.** An unknown name is silently served by the single loaded model and the reply reports `served_name`. Contrast: `/v1/embeddings` (`embeddings.zig:18–23`), `/v1/audio/speech` (`speech.zig:115–117`) and `/v1/images/generations` (`images_generations.zig:111–113`) *do* 400 with `"model {s} is not served here: send {s}"` against `srv.config.model_ids`. `server.zig:118 namedBackground` also reads `f.get("model")` (background-priority only).
- The engine never sees a model name: `api.Request` (`core/engine_api.zig:38–107`) has **no model field**. The only "does not match the loaded model" check is `Engine.submit` (`engine_api.zig:458–472`) → `error.InvalidRequest` for embedding/readout shape mismatches, mapped in `chat.zig:253`.

### Prompt build and submit
- `server/chat.zig:134 prepare` — render template→token ids (`prompt.zig`), context-window checks, `resolveSampling/resolvePenalties`, `chunk_plan.withCut` (chunked-prefill starts, `server/chunk_plan.zig`), build `api.Request`, grammar/tool structures (`srv.checkFeatures`), `engine_refusal.of(srv.engine, ...)` (`server/engine_refusal.zig` → `Engine.vtable.refusal`). Returns `Prepared{ request: api.Request, ... }`.
- `server/chat.zig:220 generate` — the **submit point**:
  ```zig
  const id = srv.next_id.fetchAdd(1, .monotonic);
  srv.engine.submit(id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| ...
  ```
  (`srv.engine` is a single `api.Engine` vtable handle field on `Server`, `server.zig:136`.)
- Engine events land in `server/mailbox.zig` (`Mailbox.onEvent`, called **on the engine thread**); the connection thread's `server/chat_generation.zig:166 Generation.loop` waits on the mailbox condvar and streams deltas through `openai.zig Run.StreamSink` → SSE. `first_ns` (`chat_generation.zig:50,96`) records time-to-first-token (`runtime.time_to_first_token`, line 291).

### /v1/models
- GET branch: `server/routes.zig:105` (`if (std.mem.endsWith(u8, route, "/models")) return status.models`).
- `server/status_routes.zig:77 models` — emits **one entry per `srv.config.model_ids` id**, all carrying the *same* `srv.info` capabilities/context window of the single loaded engine (`modelValue`, line 98). `/health` also exposes `model` + `model_ids` (`status_routes.zig:30–33`). All ids are aliases of the same weights today; an MoM fork must give each id its own `Info`.

## 2. Where a request-level ROUTER hooks in cleanly

Ranked, all before any forward pass:

1. **`openai.plan()` (server/openai.zig:129, esp. `replyModel`/`.named` at :110/:188)** — the only place the client's requested id is read in parsed form today. A router rule keyed on `body.model` is a few string compares here (well under 5 µs, not 5 ms). This is also where today's silent fallback lives, so per-model validation ("model X is not served" 400) belongs here for parity with embeddings/speech/images.
2. **`chat.generate()` (server/chat.zig:220–256), immediately before `srv.engine.submit`** — the last point before any GPU work and the *only* place `srv.engine` is touched for text generation. For MoM: carry the chosen engine handle on `chat.Prepared` (e.g. `prepared.engine`) and submit to it. `chat.prepare`'s `engine_refusal.of(srv.engine, ...)` (chat.zig:215) and `srv.checkFeatures` must consult the same chosen engine.
3. **`Server` struct (server.zig:133)** — replace the single `engine: api.Engine` + `info: api.Info` fields with a table `[N]{name, Engine, Info, Text}`; `Config.model_ids` already carries the id list (serve.zig:149–156), extend `Config` with a per-id → index map. Precedent for hot-swapping the live handle under a lock already exists: `idle_unload.park/reload` (server/idle_unload.zig:181–281) does `srv.engine = m.stub.?.engine()` / `srv.engine = opened.engine; srv.info = info` under `m.mutex`.
4. Alternative, earlier: **`routes.post()` (routes.zig:114)** — body not parsed yet; a rule router here must scan raw bytes for `"model"`. Doable < 5 ms, but duplicates parsing and would mis-key the capability gates at routes.zig:152–174, `idle_unload.wakeReload`, `srv.load.enter(group)`, `srv.idle_manager`, `status.models`/`/health`, which all assume one engine. Those are the real fork surface, not the submit call.

The Responses/Messages/Anthropic routes all funnel into `openai.runNamed` → same `plan()`, so one router in `plan()` covers chat, completions, responses and messages at once.

## 3. Scheduler / continuous batching / per-model queue split

The scheduler is the **`LaneHost`** (`core/lane_host.zig:35`): "The lane core served to the HTTP threads: one thread owns `core` and steps rounds while any stream lives."

- **Structs**: `LaneHost` (`core/lane_host.zig:35`), `LaneHost.Job` (`core/lane_job.zig:12` — owns `request: *const api.Request`, `sink`, `stream: lanes.Stream`, timing, cache entry, kv `lease`), the lanes core `lanes.Engine` (`core/lanes/engine.zig:29` — `live: std.ArrayList(*Stream)` "admitted streams in admission order (Python `_live`)", `arena`, `steps`, draft stats), `lanes.Stream` (`core/lanes/stream.zig`).
- **Queues (all `std.ArrayList`, under one `mutex` + `wake` Condition)**:
  - `LaneHost.queued` — not yet admitted; `submitFn` (lane_host.zig:199–222) inserts a new `Job`, keeping **foreground before background, each in arrival order** (line 213–215), then `h.wake.signal`.
  - `LaneHost.admitted` — jobs holding a lane; filled by `takeJob` (lane_host.zig:350) via `admit.pick` (`core/lane_host_admit.zig:11`: head-of-line with KV-pool/ledger room rules, prefix-cut and disk-read waiters passed over; `batches()`/`many()` admit several prompts in one pass when the backend has `prefill_many`).
  - `LaneHost.filling` — "admitted jobs whose prompt is still going in, **a chunk a round**, oldest first" (lane_host.zig:47). Chunked prefill is driven by `fill.run(h)` = `core/lane_host_fill.zig:150` (short prompts back-to-back, one long-prompt chunk per visit, `Share` wall-time budget `TENSORFOLD_PREFILL_SHARE`, `policyRun` fcfs variant; chunk starts come from the server-side `chunk_plan.Plan` in `api.Request.chunks`).
  - `LaneHost.cancels`, `lessons` (Sliding Weights), `scores`, plus the lone-driver path (`lane_host_lone.zig`, `api.Lone`) for a single greedy stream outside the lane core.
- **Continuous batching loop** — `LaneHost.run` (`core/lane_host.zig:504–574`), the single engine thread started by `start()` (`std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h})`, line 126):
  ```zig
  while (true) { ends.takeCancels; pressure.step; rows_host.answer; scoring.answer; learn.refuse;
      while (h.admitOne()) {}        // queued → admitted → launching (fill queue / lone / whole prompt)
      fill.run(h);                   // chunked prefill step(s)
      h.core.sweep();                // reap finished streams
      ends.deliverPending(h);        // emit tokens/finished to mailboxes
      ... if idle: h.wake.waitTimeout(100ms) ...
      h.core.step();                 // ONE decode round for every live stream
  }
  ```
  `lanes.Engine.step` (`core/lanes/engine.zig:118`) is the batched round: over `live` streams it either runs a shared round (`takeTurns`/`roundStreams` when `cfg.family_streams` and >1 live) or a per-stream pipelined/windowed round. A family round planner (`lane_rounds.zig`/`LaneHost.rounds`) can replace the `prefill_step` chunking.
- **Engine vtables**: `LaneHost.plain_vtable/scored_vtable/decided_vtable` (lane_host.zig:160–184); `LaneHost.engine()` (line 154) hands the `api.Engine` to the backend, which returns it from `open` (e.g. `native/metal.zig:274–275`: `try h.host.start(); return .{ .engine = h.host.engine(), ... }`).
- **Where a queue could be split per-model**: it already *is* one LaneHost per loaded model — each family host (`Host` struct in `native/metal.zig:89`, `native/cuda.zig:61`) embeds exactly one `core: lanes.Engine` + one `host: api.LaneHost` with its own thread and its own `queued/admitted/filling`. An MoM fork therefore splits at the **`Server` level, not inside LaneHost**: N `api.Engine` handles (each a LaneHost-backed engine), and the router's chosen handle is used at `chat.generate`'s `srv.engine.submit`. The scheduler internals need no change: per-model lanes (`Info.lanes`, `--parallel`), admission, chunked prefill and the batching round are all per-LaneHost. What is *shared* today and would need per-model treatment: `srv.next_id` (global request ids — fine to keep), `srv.preparing`, `srv.keepalive`, `srv.idle_manager` (one engine to park/reload — must become per-engine), `srv.load` ledger (attach one producer per engine), `srv.metrics`, and `admitConnection`/admission limits (per-endpoint, fine).

## 4. Concurrency: threads/tasks, singletons

- **Accept thread**: one, `serve.zig:291` → `Server.serve` (`server.zig:426`).
- **Connection threads**: one per TCP/Unix connection (`server.zig:431`, 8 MiB stack, detached; counted by `srv.open_connections`, capped by `config.max_connections` via `http_admission.zig`). Keep-alive loop runs multiple requests serially on the same thread; each request parses, prepares, submits, then blocks in `Generation.loop` reading its `Mailbox` (mutex + condvar, `server/mailbox.zig`). No async/await tasks anywhere — plain OS threads.
- **Engine thread**: exactly one per `LaneHost` (`core/lane_host.zig:126`, 16 MiB stack) — it alone owns `lanes.Engine` and runs the continuous batching loop; HTTP threads never touch the GPU. All cross-thread state goes through `LaneHost.mutex` + `wake` (`lock`/`unlock` at lane_host.zig:325–333; `unlock` also publishes the load ledger).
- **Sink callbacks** (`api.Sink`, `engine_api.zig:148`): "Called on the engine's thread: it must return at once and not re-enter the engine."
- **Global scheduler singleton?** No process-global scheduler. The scheduler is *per engine instance*: `LaneHost` lives inside the backend's host struct (`native/metal.zig:89 host: api.LaneHost`), reached via `api.Engine.vtable.ctx`. The single-engine-ness comes from `Server.engine: api.Engine` (`server.zig:136`) — one field, set at `Server.init` from `serve.Setup.engine`, mutated only by `idle_unload` under its manager mutex (`idle_unload.zig:195, 274`). Secondary threads: idle-unload timer (`idle_unload.zig` timer), keepalive ticker (`api.keepalive.Keepalive`), live status ticker (`live.zig`), grammar workers (`grammar_parallel.zig`), PDF/worker helpers — none of which touch the scheduler except `keepalive.begin/end` around submit (chat.zig:247).
- **Cancellation**: `Engine.cancel(id)` appends to `LaneHost.cancels` under the lock; `Generation` issues it via `gen.cancel()`; the engine thread applies cancels each round (`ends.takeCancels`).
- **Shutdown/drain**: `serve.zig:334 drain` — `srv.draining` flag → new POSTs get 503 (`routes.zig:141/228`), waits `TF_DRAIN_S` for `srv.generating` to hit 0, then `EngineStop.halt` (rounds end at a boundary) and `close`.

## MoM fork cheat-sheet (concrete)

| Concern | File : symbol | Change |
|---|---|---|
| requested model name, validation, reply id | `openai.zig:110 replyModel`, `plan:129/188` | resolve → validate against per-model table (400 like `speech.zig:116`), attach chosen engine to `Plan`/`chat.Input` |
| model→engine pick (last point before forward pass) | `chat.zig:249 srv.engine.submit` | submit on `prepared.engine`; also `chat.zig:215 engine_refusal.of`, `srv.checkFeatures` |
| engine table | `server.zig:133 Server.engine/info`, `serve.zig:181 Config` | N engines; `idle_unload.zig:195/274` shows the swap pattern (lock + `srv.engine = ...`) |
| N engine loads at startup | `main.zig:154 startup.both` / `engines.open` (native/metal.zig:127) | call per model; per-model `model_text.Text` too (`Server.text`) |
| /v1/models | `status_routes.zig:77 models` | per-id `Info`/`created` instead of one `srv.info` |
| scheduler | `core/lane_host.zig:35 LaneHost` (queued/admitted/filling, `run`, `lanes.Engine.step`) | untouched — one LaneHost/thread per engine already |
| per-engine idle park/reload | `idle_unload.zig` Manager (state `.resident/.unloaded/.loading`) | key manager per engine; `routes.zig:176 wakeReload` per selected engine |
