# B1 work spec — multi-model registry & loader (native fork)

Branch `mom/main` in `src/tensorfold/`. Design basis: `02-component-options.md` §1/§6,
anchors in `01-exploration-findings.md` §1. Every line ref is upstream `14dea6c6`.
Goal state: `tensorfold-native serve --model DIR_A --name A --lane trivial --pin
--model DIR_B --name B --lane default [--idle-after S]` loads both, prints a plan
line, serves `/v1/models` with both ids + per-model info, and **N=1 behaves
byte-identically to upstream**.

## New files

- `zig/src/server/models.zig` — `ModelBundle{ id, aliases, opened: api.Opened,
  info: api.Info, text: model_text.Text, lane: Lane, pinned: bool, idle_after_s }`,
  `Registry` (ArrayList + id lookup, insertion-ordered; default lane = policy
  default), `plan()` (pre-open budget check), `park/wake` stubs (B5 fills them).
- `zig/src/server/mom_cli.zig` — parse repeated `--model/--name/--lane/--pin/
  --idle-after` (extend, don't break: `server/cli.zig parse` stays the single-model
  path when ≤1 positional present).

## Touch-points (keep each ≤ ~30 lines; rebase-friendly)

| File | Change |
|---|---|
| `server/main.zig:71-190` | loop over parsed models: `hub.resolve` each, `startup.both` per model, collect `[]api.Opened`; `mom` gate: >1 model → `server/mom.zig bootstrap` (new) else existing path untouched. Refuse: TP args, device pinning >1 device, `--slide` (v1). |
| `server/serve.zig:134-229` | `run(...)` accepts `[]ModelBundle` (single-model call sites pass a 1-slice); builds `Server` with `models` table; `config.model_ids` = union of all bundles' ids (deduped). |
| `server/server.zig:133-192` | add `models: []ModelBundle` + `default_model: usize`; **keep `engine`, `info`, `text` as aliases of `models[default]`** so all untouched readers keep working at N=1 and single-lane defaults elsewhere. `slide`/`recall`/`voices` stay from default bundle (v1). |
| `native/cuda.zig:155 open` | (1) accept explicit `memory_quota_gib` (0 = auto = today) threaded from the plan; (2) `g.ctx` ownership: register with a per-process `ContextOwner` refcount instead of assuming sole owner. |
| `zig/src/cuda/context.zig:23,57-63` | refcount the primary context per owner; `deinit` only releases when last owner leaves; `copy_source.dropSpares` only then; device-sync only when no other owners (else stream-sync). Unit-testable shape: pure refcount struct. |
| `zig/src/cuda/memory.zig:11-21` | add per-owner quota ledger: `Owner{id, soft_cap}` under the existing mutex; `usage(owner)`; admission (`native/cuda_memory.zig budget.admit`) consults min(global room, owner quota). Global limit semantics unchanged for N=1. |
| `zig/src/cuda/graph_registry.zig:23` | 8 slots → `8 * max(1, owners)` (grow on register; report-only consumer). |
| `zig/src/cuda/copy_source_budget.zig` + `copy_source_capture.zig` | per-owner sub-budget (split evenly at register); host test: two concurrent captures don't starve each other. |
| `core/load_ledger.zig:104-107` | allow one producer per model id (key producer by bundle id). |
| `server/status_routes.zig:77 models` | one entry per bundle id with **that bundle's** `info`; `/health` adds `models:[{id,lane,state,parked}]`; keep top-level `model` = default id (client compat). |

## Plan-time budget (server/models.zig plan())

sum per model: `native/cuda_family_memory.weights` estimate + streams×window bytes
(from family sizing) + fixed slack; + ONE process reserve
(`native/cuda_memory.zig reserveBytes`); vs measured free. Over budget → refuse at
startup naming the dominating term. Quota per model = its plan share (hard cap for
its pools). Print `mom plan: A weights=1.6GiB kv=1.4GiB quota=16GiB; B ...` line.

## Explicitly NOT in B1 (staged later)

Router (B2 — `Server.engine` alias means B1 still serves everything on the default
bundle: correct interim behavior); KV tag-fold (B4); park/wake real bodies (B5);
per-model warm/idle bookkeeping review (B3); idle_unload multi-target (B5).

## Acceptance (B1 exit)

1. `zig build test` green (incl. new host tests: plan refuse/pass, context refcount,
   quota ledger, graph-registry growth).
2. N=1 parity: upstream test suite + `tools/exact.py`-style drafted==plain on small
   lane == pre-fork binary results (same sha fields).
3. Two-model device run: capacity lines ×2, `/v1/models` shows both ids with
   distinct capability blobs, explicit `model:B` on chat answers from B (B2 not yet
   in: via the default-bundle alias this means explicit still answers default —
   acceptable interim, logged); `over-budget pair → startup refusal` on a constrained
   cgroup or env-capped free.
4. Memory plan line matches `nvidia-smi`/RSS within 10%.
