# TensorFold Zig engine: one-model serving path and what stands between us and N engines in one process

Read-only exploration of `/home/derek/tensorfold-mom/src/tensorfold/zig` (branch tip `14dea6c`). All paths below are relative to `zig/src/` unless noted.

## TL;DR

The engine is already built as trait-object instances (`api.Engine` = ctx+vtable, `native.Host` heap-allocated per model, `LaneHost` with its own thread). Nothing in the load path is structurally single-instance — the "one model per process" property is enforced *above* the engine (`server/main.zig` opens exactly one) and by a handful of process-global CUDA/accounting singletons. The codebase **already keeps 2–4 neural networks resident in one process** (target + drafter + vision tower + Sliding-Weights trainer) behind one CUDA primary context. Two `Host` instances in one process is feasible; the work is mostly in `server/`, not the engine core.

---

## 1. The `serve` path today

- **`main.zig` (repo entry, Metal one-shot `run`)** — `main.zig:274-279` delegates unknown commands to `cluster/cli.zig` (`serve --cluster`, disaggregated MCDMA nodes) and `cli/cli.zig`. `cli/cli.zig:22-25` is only `models|info|pull|merge|plan` (checkpoint subcommands); it explicitly does *not* take `serve` (test at `cli/cli.zig:90`).
- **The serve binary is `tensorfold-native`, rooted at `server/main.zig`** (`zig/build/metal_targets.zig:67-71` `nativeServer`, and the CUDA analog wires module `native_engines` → `native/cuda.zig` in `zig/build/cuda.zig:16-46`).
- Flow in `server/main.zig`:
  1. `argv[1] == "serve"` (`:71`, `:98`) → `server/cli.zig parse` (`:104`) → one positional `model`.
  2. `hub.resolve` model dir (`:124`), optional `--drafter` resolved like a second model (`:127`).
  3. `modelType` from `config.json` (`:128`, `:229`).
  4. rank>0: `engines.follow` (`:141-144`), no HTTP.
  5. `startup.both(loadText, engines.open)` (`:154`; `server/startup.zig:12` runs tokenizer load on a second thread) → **exactly one** `api.Opened`.
  6. `serve.run(gpa, io, args, .{ .engine = up.engine.engine, ... })` (`:174-190`; `server/serve.zig:134`), which does `Server.init(gpa, io, s.engine, s.text, ...)` (`serve.zig:229`).
- **`server/engines.zig:45 open`** forwards to `native.open` — the build-time module (`native/cuda.zig` for CUDA builds).
- **`native/cuda.zig:155 open`** dispatches: special-cases `fish_omni`/`qwen4*`, else `inline for (registry)` (`:166`) over the compile-time family list `native/cuda_families.zig:15` (`nemotron.native, glm.native, qwen35.native, …`).
- **`native/cuda.zig:177 openWith`** is the whole single-model story: per-model `cuda.Driver.open()` + `cuda.Context.init` (`:227`, `:238`), memory-pool admission (`:256`, `family_memory.weights` `:258`), `F.open(gpa, io, &g.ctx, o.dir, kernels, opts)` (`:282`), drafter wrap (`:336`), stream counting (`:353`), `gpa.create(Host)` (`:365`), `h.serve(...)` (`:445`, starts the lane thread), returns `api.Opened` (`:474-476`).
- Rank>0 follower: `native/cuda_follow.zig` (`follow` re-exported at `native/cuda.zig:12`).

## 2. Engine structs and checkpoint load

- **`core/engine_api.zig:421 Engine`** — `ctx: *anyopaque` + `vtable` (info/submit/cancel/status/memory/learn/recall/…). Pure instance data, no globals; `Opened` (`:372`) wraps engine+close+halt; `Open` (`:277`) is the server's request to open one checkpoint.
- **`native/cuda.zig:51 Host`** — "One loaded model behind the lane host": holds `gpu: ?*Gpu`, `family: *anyopaque` (the family's Loaded), `inner` backend, `cfg`, `core: lanes.Engine`, `host: api.LaneHost`, KV ledger, prompt store, slide learner, recording, and `served` (target+drafter wrapper). Heap-allocated per open.
- **`core/lane_host.zig`** — the served lane core; `thread: ?std.Thread` (`:53`) spawned in `start` (`:126`), one engine thread per host; `on_thread` (`~:86`) is the hook where a CUDA family binds its context on that thread (`native/cuda.zig:99 initThread`, `:91 bind`).
- **Family engine (nemotron example)** — `families/nemotron/cuda_engine.zig:70 Engine`: own `ctx`, `stream`/`side` streams (`:145`), `k: Kernels` loaded per engine (`:153`, `kern.Kernels.load` — each engine loads its own fatbin modules), `w: weights.Weights` (`:184` `weights.load` = the checkpoint load), `b: state.Buffers`, graphs, pinned buffers, `vision` (`:196`), `tp` group (`:163`). Failed loads park on a global registry: `load_owner.keep` (`families/nemotron/cuda_load_owner.zig:5` linked list).
- **Checkpoint loading** — `core/checkpoint.zig:65 Checkpoint` (safetensors namespace, `openModel` `:82`); shard/prefetch/digest helpers `checkpoint_index.zig`, `checkpoint_prefetch.zig`, `digest_cache.zig` (on-disk keyed cache, 1024 entries — content-keyed, model-count-agnostic); EXL3/EXLR8 loaders `exl3_load.zig`/`exlr8*.zig`. `core/checkpoint_drop.zig:92 var pending` is a process-global page-cache-advice queue, only active under TP. CUDA admission estimate: `native/cuda_family_memory.zig:46 weights` reads tensor sizes from the folder.

## 3. Process-global state inventory (what a 2nd engine would share)

| Global | Location | Multi-model risk |
|---|---|---|
| Device/host byte accounting, single `allocation_limit`, single peak | `cuda/memory.zig:11-18`, `usage()` `:48`, `setAllocationLimit` `:21` | Medium: one cap/peak for all engines; admission uses deltas so it mostly copes |
| Graph-cache report slots, fixed 8 | `cuda/graph_registry.zig:23` (claimed at `cuda/graph_cache.zig:148`) | Low: >8 caches just stop being reported in /health |
| CUDA graph/copy-source capture scopes (64) + global copy budget | `cuda/copy_source_capture.zig:7-8`, `cuda/copy_source_budget.zig:6-9` | Medium: budgets shared across engines; scopes fine until 64 |
| Context health table (64 entries keyed by ctx) | `cuda/context_health.zig:6` | Low: multi-context capable |
| Retained failed `Gpu` list | `native/cuda_load_gpu.zig:5-7` (linked list, `Gpu{driver,ctx}` `:5`) | None: designed for several |
| Retained failed `Engine` list (`load_owner`) | `families/nemotron/cuda_load_owner.zig:5-6,29` | None: linked list |
| TP watchdog (rank sockets, `my_rank`, peer digests) | `cuda/tp_watch.zig:20-29`; used from `native/cuda.zig:283`, `cuda/completion.zig`, `cuda/nccl.zig:115` | High **only for TP models**: one rank group per process |
| thread-local bound context per engine module | `native/cuda.zig:47`, set in `bind` `:91-98` | Low: each engine's own lane thread binds its own; HTTP threads never bind |
| Host-cache peak probe | `native/cache_fit.zig:62` | Low: reporting only |
| Env-var caches (`TF_COOP_SPLIT`) | `families/nemotron/forward.zig:440` | None: read-once config |
| Server stop flags / signals | `server/serve.zig:20-24` | None for N engines (process shutdown) |
| Grammar lib (dlopen'd once) | `core/lanes/grammar.zig:26-28` | None |
| Per-`Ledger` single engine producer | `core/load_ledger.zig:104-107` ("one producer at a time") | Medium if one Server fronts N engines; fine if each engine keeps its own ledger |
| `src/gpu_memory.zig` | 4-line `fits()` helper only — **no** allocation state lives here | — |

CUDA context itself: `cuda/context.zig:23 init` retains the device **primary context** (driver-refcounted; two engines on one GPU share the same handle). Caveat: `Context.deinit` (`:57-63`) does a device-wide `synchronize()` and `copy_source.dropSpares(d, handle)` — closing engine A while engine B runs would drop B's spares and stall it. `setSchedule` (`:18`) is ignored once the primary context is active (engine 2 can't pick a different schedule).

## 4. >1 neural network in one process — already happens

- **Speculative drafter (`--drafter`)**: `native/cuda.zig:336 drafted.wrap` → `native/cuda_drafted.zig:52-89`; family hook `F.openDrafter` (`families/qwen3_5/cuda_drafter_open.zig:21`: DFlash2 `Draft.init` `:30` or the MoE's **MTP layer** `:48`; also `families/gemma4/cuda_native.zig:279`). The draft model is a full second net — own `Config`, `weights.Weights`, `cuda.Module`, kernels, device buffers (`families/qwen3_5/cuda_draft.zig:49-72`) — held as `l.draft`/`Held` beside the target and stitched in by `lanes.drafted.Drafted` wrapping the target's backend vtable. Shares the target's CUDA context and streams.
- **Vision tower**: `e.vision = cuda_vision.Vision.init(...)` (`families/nemotron/cuda_engine.zig:196`) — second network on the same driver, held per engine. RADIO tower (`families/radio/tower.zig:27`) keeps its own failed-load registry (`families/radio/weights.zig:278`).
- **Living/Sliding Weights trainer**: `native/cuda.zig:412-427` (`slide_adapter.Adapter(F.slide)`, `F.learnModel`); `families/nemotron/cuda_train.zig:75 Trainer` — a training net with its own `state.Buffers`, raw dequantized weights, head_t, gradient buffers, `learn_bytes` budget (`native/cuda.zig:342`). Family-neutral learner in `living/` (`living/living.zig`: pockets, shards, router). Runs on the same context via the engine's `trainer_slot` latch (`cuda_engine.zig:105 slots`).
- So per serve: target + drafter/MTP + vision + trainer are already 2–4 networks under one primary context — the machinery for co-resident nets exists; it is just always *one target*.

## 5. Single-model assumptions a multi-model fork must change

1. **`server/main.zig:71-190`** — one positional model, one `hub.resolve`, one `startup.both(…, engines.open)`, one `serve.run{.engine}`. Add N opens + per-model tokenizers + router. *Large but mechanical.*
2. **`server/server.zig:133-192`** — `Server` holds one `engine: api.Engine`, one `info`, one `text` (tokenizer/eos/think markers), one `slide` graph, one `recall`. Needs per-model bundles. *Core of the fork.*
3. **`server/serve.zig:82 Setup.engine` / `:229 Server.init`** — one engine + `Closer`/`Reopen` (`idle_unload_closer.zig`). Route by requested model instead. *Straightforward.*
4. **`native/cuda.zig:47,91-98` `bound` threadlocal** — per-thread single context is fine for N dedicated lane threads, but any shared-thread CUDA call must `makeCurrent` per call. *Already correct via bind().*
5. **`cuda/memory.zig:11-18,21`** — one device-byte counter, one peak, **one `allocation_limit`**; admission (`native/cuda.zig:256-281`) checks the whole process. Give each engine a quota or accept shared cap. *Medium.*
6. **`cuda/context.zig:57-63`** — `Context.deinit` device-synchronizes and drops shared copy-source spares; closing engine A harms running B. Refcount spares per owner. *Small, real hazard.*
7. **`cuda/graph_registry.zig:23`** — 8 report slots total; N engines × caches overflow. Raise cap. *Trivial.*
8. **`cuda/copy_source_budget.zig:6-9`, `copy_source_capture.zig:7`** — shared global copy-capture budget and 64 scopes; concurrent capture from two lane threads untested. *Medium; needs a test.*
9. **`cuda/tp_watch.zig:20-29`** — process-singleton TP rank watcher: cannot host two TP-TP models, or mix TP + non-TP, in one process. *Exclude TP models from v1 MoM.*
10. **`core/load_ledger.zig:104-107`** — one producer per Ledger; one Server-ledger per engine, or N producers. *Small.*
11. **`native/cuda.zig:248` `TENSORFOLD_CUDA_KERNELS`** and all `TF_*`/`TENSORFOLD_*` env knobs — process-wide; both engines get the same schedule/kernels. *Accept or per-model config.*
12. **`server/main.zig:134-139 originals.serving`** — folder hold is per-dir, fine; `--idle-unload` policy (`server/idle_unload.zig`, `idle_unload_policy.zig`) assumes one parked engine per server. *Rework per-engine.*
13. **`native/cuda.zig:51 Host` + `core/lane_host.zig:126`** — no change needed: already per-engine objects with own threads. *Ready.*

## 6. Feasibility: 2 engine instances in one process

- **Verdict: high.** Engines are instance structs (`api.Engine` vtable, heap `Host`, per-host `LaneHost` thread); process already runs 2–4 networks per model.
- Cleanest fork point: call `native.open` N times (it never assumes "first engine"; `Gpu`/`load_owner` registries are lists), then a router that maps request→`Opened.engine`.
- Must-fix before two CUDA engines coexist: per-owner copy-source/context deinit (#6), shared allocation cap (#5), graph-registry slots (#7), concurrent stream-capture budget (#8).
- Avoid in v1: TP models (`tp_watch` singleton), `--slide` trainer + second engine on same GPU only via its `learn_bytes` admission (safe but unbudgeted across engines).
- `--idle-unload` (`server/idle_unload.zig:210 reload`) already opens/closes engines at runtime in-process — proves dynamic load/unload works; MoM keeps them resident instead of swapping.
- Second GPU is free: `api.Open.device` is per-open (`native/cuda.zig:207`), each `Gpu` retains its device's primary context.
- Metal/HIP/XPU builds use the same `server/main.zig` with a different `native_engines` module; the same router applies, but this report's CUDA-specifics (contexts, graph capture, tp_watch) are CUDA-only.
