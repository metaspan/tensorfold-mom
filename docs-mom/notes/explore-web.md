# TensorFold Web — Python layer exploration for a Mixture-of-Models front door

Read-only pass over `/home/derek/tensorfold-mom/src/tensorfold/web/tensorfold_web/` (all paths below relative to that package unless noted).

## 0. Shape of the layer (one paragraph)

`cli.py:main()` holds a per-user file lock (`instance.py:hold()`, `web.lock` — **exactly one TensorFold Web per user, by design**), loads one `Settings` (`settings.py:load()`), builds **one** `Engine` and **one** FastAPI app via `server.py:create_app(settings, engine, ...)`. `create_app` then wires ~15 `server_*.install()` modules around those singletons. The public OpenAI-compatible surface is the relay mounted at `server_relay.py:install() → @app.api_route("/engine/{path:path}")`, which delegates to `proxy.py:Proxy.handle()`.

## 1. Multiple engines behind one OpenAI endpoint — today? **No. One engine, one URL, one relay.**

- **Single engine:** `server.py:create_app()` line ~91: `eng = engine or Engine(s, state_dir=...)`; `cli.py` likewise. Every consumer (`server_engine.py`, `bench`, `history`, `usage`, `teacher`, `series/switch`) closes over that one `eng`.
- **Single address/key:** `Settings.engine_url` + `Settings.api_key` (`settings.py`); `Settings.engine_port()` is the *one* managed port. `Engine.start()` raises `"an engine is already running ({self.model}); stop it first"` (`engine.py:226-229`).
- **Where the proxy is:** `proxy.py:Proxy` (73 lines). `Proxy.__init__(engine_url_getter, api_key_getter, target)` — target is read **once per request** as `(url, key)` (`server.py:99`: `Proxy(lambda: s.engine_url, eng.key_in_use, target=lambda: eng.target()[1:])`). It strips hop headers, injects `Authorization: Bearer <engine key>`, sets loop guard `x-tfs-relayed` (`proxy.py:RELAYED`), caps body at 64 MiB, and streams the upstream response raw (`StreamingResponse`, 600 s timeout). Route registration + loop/path policy: `server_relay.py:relay()` + `server_http.py:relay_refuses()` (blocks `slide`/teach routes, `?#\..` tricks).
- **How does the `model` field select an instance?** **It doesn't — the Web layer never reads it.** `Proxy.handle()` forwards the body as opaque bytes; whatever `model` the client sent goes verbatim to the *single* engine, which answers however it likes (the native engine serves the one checkpoint it was started with; `mock_engine.py` ignores it). The UI itself sends `model` in `POST /engine/v1/chat/completions` (`ui/src/screens/Chat.tsx:237,326`, `ChatParallel.tsx:49`) but that's client-side labeling. There is **no `/v1/models` route on Web itself** — clients reach `/engine/v1/models`, relayed to the single engine.
- **Closest existing "multi":**
  - `fleet.py` + `server_fleet.py`: register up to 16 *peer TensorFold Webs* (`Peers`, `fleet.json`); `fleet.engine_of(peer)` returns `(peer_url + "/engine", key)` — i.e. engines on *other machines* are reached through *their* relay. One target per call, no mux, no public endpoint merging them.
  - `server_bench.py` / `server.py:engine_named(spec)`: resolves a bench/KL/replay target from `{"url", "key"}` or `{"peer"}` via `engines.identify()` — a **multi-target addressing seed** already in the codebase.
  - `replay.py:Replays._start()` accepts `target={"kind":"address","url","key","model"}` — replay a conversation against an arbitrary OpenAI-compatible server.

**Bottom line:** one endpoint, one engine. If one *upstream* server hosts many models, `/engine/v1` already behaves like a multi-model OpenAI endpoint — but the Web layer manages exactly one local child.

## 2. Instance lifecycle (spawn / health / restart)

- **Spawn:** `engine.py:Engine.start(model, native_model=..., drafter=..., slide=..., serve=...)` → `engines.py:choose(native_model)` → `NativeLauncher`/`PythonLauncher` → `native.py:build_command()` (native binary + `serve <model> --host --port ...`, flags gated by `capabilities --json`, `serve_flags`, `safe_extra`/`check_extra`) or `engine.py:build_command()` (Python line: `tensorfold serve ...`). Launched with `subprocess.Popen(..., start_new_session=True)`, stdout+stderr to a **file** (`engine.json`/`engine.log` under `state_dir`), log tailed by `_pump()` thread into a 6000-line deque with subscribers for the SSE log stream (`server_engine.py:engine_logs_stream`). Key goes via env `TENSORFOLD_API_KEY`, never argv (`engine.py:107`).
- **Adoption across Web restarts:** `Engine._save_state()` writes `engine.json` (pid, create_time, model, kind, command, active_key, started_at, log path); `Engine.adopt()` re-adopts a live child via psutil `create_time` match. **One `engine.json` per `state_dir`** — N engines in one Web need N subdirs (trivially parameterized, see §4).
- **Health:** `Engine.health()` polls `/health`, falls back/augments with `/v1/load`, `/v1/models`, `/metrics` (`engine_metrics.py:parse_health/parse_load/parse_metrics`); `status()` layers process state (`starting/failed/stopped`), `info()` adds command/uptime/startup numbers. There is a `generation`/`target_lock` scheme so readings never leak across an address/key change. Periodic sampling exists: `history.py:History._loop` polls `eng.info()` every 5 s (Monitor charts + `usage.py:Usage.record`).
- **Restart:** **none automatic.** Stop is `Engine.stop()` (SIGTERM → wait 20 s → kill). Starts are driven by `server_engine.py POST /api/engine/start` → `server_engine_control.py:_start_held` (settings patch, `start_lock`, `gate`) → `launch()` which *does* contain one policy-like fallback: native-qualification refusal → `native.remember_refusal()` → restart on the Python line (`server_engine_control.py:launch`, `native.refusal_in`). `series.py:switch` (Compare) does scripted stop→start→wait-running with a 15-min deadline — reusable "start and wait" logic. `health.py:Checker.check()` turns readings into findings (killed/OOM, KV full, waiting…) but nothing acts on them.
- **Serialization:** `gate.py:Gate` — one global reservation for "everything that changes the engine or its files"; `server.py` also has `start_lock` (asyncio) and `settings_lock`.

## 3. Routing policy layer? **None.** Pure single-target pass-through.

There is no rules/stickiness/fallback layer anywhere on the request path. The only "routing" facts of life: (a) the relay's target getter (one `(url, key)` pair), (b) `relay_refuses()` path policy, (c) `Proxy`'s loop guard. No retry-on-error (502 on unreachable, `proxy.py:60`), no per-model table, no queue, no stickiness. Model→instance mapping doesn't exist because instance-set = 1.

## 4. Adding the MoM front door *in this layer* (v1: multiplexed child processes, one Python process)

Good news: **`Engine` is already instance-parameterized** (`Engine(settings, state_dir)`, each with its own `Settings` → own `engine_url`/port/key, own `_Child`, own adopt-state file). The blockers are purely the `create_app`/cli singleton wiring and shared services. Nothing in `engine.py`, `engine_child.py`, `engines.py`, `native.py` assumes there is only one.

### Files/functions to extend

1. **`instances.py` (new; model on `fleet.py:Peers`)** — JSON registry of instances `{id, model, engine_url, key, role: small|big|…, enabled}` in `config_dir()/instances.json` (`settings.py:write_private`, `threading.Lock` pattern straight from `fleet.py`). Plus a `Settings`-per-instance factory (dataclass replace from the user's base `Settings`, distinct ports).
2. **`server.py:create_app`** — build `Dict[instance_id, Engine]` + `Proxy`-per-instance (or one Proxy with a per-request target — its `target` callable is *already* the seam, just per-request rather than global). Keep `app.state.engine` = primary for the existing UI; add `app.state.instances`, `app.state.router`. Give each Engine its own `state_dir` subdir (`config_dir()/engines/<id>/`) so `engine.json`/`engine.log` don't collide. Give each instance its own `Gate` (or key the gate by instance).
3. **`router.py` (new; the whole point)** — pure functions, testable like `health.py`'s rules:
   - `RouteDecision resolve(body_model, body, headers, healths) -> (instance_id | None, reason)`. Rules: explicit `model` wins (exact id, then alias match against catalog `Entry.id`/`name`); else trivial-traffic heuristic (char/token length of last user turn, message count, presence of `tools`/images/system-heavy → big model; short + no tools → small hot model); else default big.
   - **Stickiness:** keep an in-process LRU `conv_key → instance` (and mirror into the conversation doc, see reuse below); `conv_key` = client header (e.g. `x-tensorfold-conversation`) or free fallback: hash of (system + first user message) — "free re-routing" needs no client change.
   - **Fallback:** if the chosen instance's `health().ok` is False or the upstream connect fails, walk `fallback` order.
4. **`server_relay.py:relay()`** — before `proxy.handle()`: for `v1/chat/completions|completions|messages|responses`, `Proxy.handle` *already buffers the full body* (`proxy.py:51-55`), so parse `model` + last messages there (or pass buffered bytes to `resolve`), consult `Router`, then call the target instance's proxy. Everything else (GET `/v1/models`, `/health`…) is answered/merged by the front door itself: merge `engines.identify(url,key).models` across instances into one `/v1/models` (`engines.py:identify` does the per-server probe already).
5. **`proxy.py:Proxy`** — accept `target=(url,key)` per `handle()` call (one-line signature change; the snapshot-per-request discipline at lines 29-30/46 was written exactly for this — "a change of engine midway never sends one's key to the other"). Keep the `RELAYED` loop guard and hop-header logic untouched.
6. **`server_security.py:_gate` + `server.py:_engine_key_given`** — auth currently compares the bearer against **one** engine's key (`eng.key_in_use()`, `s.api_key`). Needs the set of all instance keys (or one front-door key mapped per instance). Keep the "API-key header ⇒ API tool, exempt from CSRF checks" behavior (`server_security.py:40`).
7. **`server_engine_control.py` / `server_engine.py`** — reuse `_start_held`/`launch` per instance (extend `launch(model, entry)` to take the instance's `Engine` + `Settings` instead of closing over `eng`/`s`); add `/api/instances` CRUD + start/stop per instance; adopt each Engine at boot. Reuse the native→Python fallback and `server.py:switch`'s wait-for-running loop as the router's "ensure up" path.
8. **UI (optional v1)** — `ui/src/screens/Chat.tsx` already sends `model`; adding the conversation-id header is a few lines in the `fetch` at `Chat.tsx:237/326`.

### What already exists to reuse

- **`conversations.py:Conversations.update()`** — the `settings` dict **merges** rather than replaces (`lines 163-165`: "a page saving its own keeps what a replay recorded"). This is the designed extension point for persistence: store `settings.routed_to = {model/instance, at, reason}` per conversation, exactly as `replay.py` persists `settings.replay_of/target` (`replay.py:64`). Stickiness survives restarts for free; `page()` already surfaces `target`/`model` per conversation.
- **`engines.py:identify()` / `Endpoint`** — probe + merge `/v1/models`, engine/version detection.
- **`fleet.py:Peers` / `engine_of`** — the token-file/JSON-store + remote-relay addressing template (and a reminder that `/engine`-behind-`/engine` loops are already guarded via `RELAYED`).
- **`engine_metrics.py:parse_health`, `Engine.health()/status()`** — per-instance liveness probes; `History`'s 5 s loop as the probe heartbeat (currently single-engine — key rows by instance if kept).
- **`usage.py:Usage`** — already keyed `(hour, model)` with per-server baselines (`totals@{url}`, `started_at` identity): per-model traffic stats for the two tiers come free once each instance is polled.
- **`replay.py` `{"kind":"address",...}` targets and `kl.py:Target`** — "send the same turn to the big model" (free re-route/escalation, A/B) is already the shape of replay/KL calls.
- **`native.build_command` / `engines.Launcher` / `engine_child._Child`** — unmodified: each instance is a stock `tensorfold-native serve` child, distinct `--port` via its own `Settings.engine_port()`.
- **`gate.py:Gate`** — per-instance instance, unchanged class.

### Gaps vs the single-process goal

- **Singleton wiring:** `create_app`, `cli.main`, and every `server_*.install` that takes `eng` (`server_engine.py`, `server_engine_control.build`, `bench_target`, `history`, `teacher`, `living`, `series`) assume one engine. v1 should instantiate N Engines + N proxies and leave the Monitor/Bench/Teach UI pointed at a "primary" instance (or hide those screens); the request path only needs relay→router→proxy.
- **Shared state paths:** `Engine._state_path()` = `state_dir/"engine.json"` — fixed name; needs per-instance dirs (already parameterized via constructor).
- **`instance.py:hold()`** single-Web lock is *fine* for the front door (keeps it one process) but `secure_config_dir()/web.lock` must not be taken per child.
- **Memory budgeting is single-engine thinking:** `native.py:336` ("the native engine sizes its memory itself"), `engine.memory_limit_env` (one budget fraction), `fit.NATIVE_COPIES` in `catalog.py:verdict_of`, and `health.py` headroom findings all assume one resident engine. Two native children on one Mac/Spark can jointly overcommit; v1 needs an explicit budget split (catalog `verdict_of` per instance, or refuse to start the second when `machine_memory().headroom != "ok"`).
- **No failover after first byte:** `Proxy.handle` streams raw; re-routing is only possible before the upstream response starts (connect error / status ≥ 500 on the response head). True mid-stream re-route belongs to a later fork.
- **No policy config today:** nothing like `routing.json` exists; `Settings.validate()` ignores unknown fields, so a new file + API (Peers pattern) is cleaner than extending `Settings`.
- **`/v1/models` synthesis and Anthropic/Responses parity:** the relay today forwards whatever path; the front door must answer `/v1/models` itself and keep `model` rewriting consistent across `chat/completions`, `messages`, `responses` (all three are exercised in `recipes.py:run_check`).
- **Auth key fan-out** (`_engine_key_given`) and Connect/recipes (`server_connect.py:connect_ctx`, `recipes.Ctx`) are per-engine-shaped; the front door needs its own URL/key + model table in recipes.

## 5. One-line answers

1. **No** — one `Engine`, one `engine_url`, one `/engine` relay (`server.py:99`, `server_relay.py`, `proxy.py`); the `model` field is never read by Web, it's forwarded to the single engine. Nearest multi: fleet peers (`fleet.py`) and ad-hoc `{url,key}` targets (`server.py:engine_named`, `replay.py`).
2. Spawn via `Engine.start → engines.choose → native.build_command` (detached child, log file, `engine.json` + `adopt()` across restarts); health via `Engine.health()/status()`; **no auto-restart** — manual `POST /api/engine/start|stop` gated by `Gate`, with one scripted fallback (native→Python) in `server_engine_control.launch`.
3. **No policy layer at all** — pure single-target pass-through with path-refusal (`relay_refuses`) and loop-guard.
4. New `instances.py` + `router.py`; extend `create_app` (dict of Engines), `server_relay.relay` (inspect buffered body, resolve, dispatch), `Proxy.handle` (per-call target — seam already exists), `_engine_key_given` (key set), `/api/instances` CRUD reusing `server_engine_control.launch`; persist stickiness via `conversations.Conversations.update` settings-merge (as `replay` does), reuse `engines.identify` for a merged `/v1/models` and `Usage` for per-model traffic.
5. Gaps: singleton wiring + shared services (Monitor/Bench/Teach), fixed `engine.json` name, no shared memory budget for N native children, no post-first-byte failover, auth key fan-out, no routing config surface.
