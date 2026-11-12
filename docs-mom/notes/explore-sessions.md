# TensorFold KV-cache / session subsystem — exploration notes (read-only)

Repo root explored: `/home/derek/tensorfold-mom/src/tensorfold` (all paths below relative to `zig/src/`).
Primary engine exercising the sessions subsystem: `families/deepseek_v41/` (dsv41). The generic native
server uses a *different* pool (`core/kv_pool.zig`) and prompt cache (`core/prompt_cache.zig`); both are
covered because MoM will collide with one or the other per engine family.

---

## 1. How conversations/sessions are identified and keyed

**There is no conversation id anywhere in the session layer.** Identity is (a) the token sequence itself,
(b) a small `u32` tag, (c) a per-stream *slot* as execution scratch.

- `sessions/store.zig` — `Store.find(tag, prompt)` / `store_ram.save(slot, bounded, ids, tag, kind)`:
  the only lookup keys are `tag: u32` + the prompt token ids. `Extra` carries `kind` (0 = turn-end entry,
  1 = prompt replay point — see `sessions_gpu.zig:407` `prompt_kind`), `used` clock, `pages`, `bounded`,
  `file`, `origin`, `covered_by`.
- `sessions/prefix.zig` — `Index(Extra)`: a trie over *chained Blake2b-128 digests of full pages of token
  ids* under a tag: `root(tag)` (`"tf-sess-root" || tag`), `step(prev, page_ids)`, and the entry identity
  `entryKey(node, pos, tail)` (`"tf-sess-entry" || node || pos || tail`). `Entry` = `{key, tag, pos, node,
  tail, ram, extra}`. A "conversation"/chat is implicit: `isLeaf(id)` = "a RAM entry no other RAM entry
  extends: a chat's newest state"; `members(leaf,...)` enumerates a chat's older turns; `extensions`,
  `covered` define the lineage. Recency = `Store.clock` (`touch`).
- `tag` is **not a model id**: dsv41 derives it from the CED prefill mode only —
  `families/deepseek_v41/ced.zig:15-17` `tag_fast = 1<<8` (fast prefill, prompt snapshots),
  `tag = tag_fast | 1<<4` (replay); `sessions_gpu.zig:409-411` `fn tag()` picks one. Purpose: "replay
  entries never resume full ones, nor full ones replay's." It is a *mode/shape* namespace, family-defined.
- Slot = the per-stream execution handle: `sessions/pool.zig` `Pool.newSlot(capacity)`; dsv41 creates one
  `Slot` per stream (`cuda_kv.zig:144-147`, `o.slots`), keeps per-slot committed token history in
  `Sessions.hist[slot]` (`sessions_gpu.zig:100-101`), and one active slot (`kx.slot`, `SlotSet` for several).
- HTTP layer keys nothing either: the generic prompt cache (`core/prompt_cache.zig`, header: "Exact prompt
  reuse: states are keyed by tokens and canonical decoded intervals through their position") matches
  entries by token prefix; learned-state keys are `Wyhash(0x1e47, tokens)`
  (`core/prompt_imprint.zig:57-58`, `core/prompt_cache.zig:368` `sharedKey`). "Conversation" appears only
  as a policy notion (`keep_others`, `core/prompt_cache_others.zig`: evict only your own lineage).
- **Model identity appears only on the NVMe tier**, as the `compat` digest (§2/§3), never inside a RAM key.

## 2. Prefix caching: cache key and invalidation

Key construction (`sessions/prefix.zig`):
- Page digests chain: `out[j] = blake2b128(prev || ids[j*page .. (j+1)*page])`, seeded by `root(tag)`.
  One trie node per distinct full-page run; entries hang off nodes with their partial tail.
- Lookup `find(tag, prompt, digests, tier)`: binary-search the deepest trie node on the prompt's chain,
  walk up to parents; candidate entry must *strictly* prefix the prompt (`tailOf`: entry shorter than
  prompt and its tail token-equal there). `Tier.ram` restricts to RAM entries.
- `Store.find` returns `Hit{id, ram, pos}`; `historyOf` re-verifies disk entries by recomputing
  `entryKey` from the prompt's prefix (`error.NotTheEntry`).

Invalidation / eviction paths:
- Pool refcounting (`sessions/pool.zig`): pages have `refs` (slots + entries + jobs) and `slot_refs`; an
  entry alone holding pages makes them `entry_only` (parkable). `Pool.drop` → page back to `FreeSet`.
- `Store.leaveRam` drops pages+bounded state; the entry stays indexed iff a disk file holds it
  (`extra.file`), else `index.remove` (which prunes trie nodes with `sub_all == 0`).
- Budgets (`sessions/store.zig` `Policy`): `ram_bytes` (bounded-state budget, LRU evict in `trimRam`),
  `chats` + `resident_bytes` (idle-chat count / `pool.entryOnlyBytes()` caps enforced by `enforce()`,
  leaf parked, unshared older turns dropped; `park_prefixes` opt-in), admission-driven spill via
  `planSpill(need, keep, out)` (coldest-first, never `keep`, counts pages whose refs hit 0).
- `skip_covered`: an entry a newer entry of its chain covers is dropped, not parked (`covered_by` set in
  `settle` of a park; `Index.covered`). Disk trim (`Disk.trim`) deletes LRU `.tfs` files over budget;
  `Store.trimDisk` unindexes entries whose files died. `forget()` handles unreadable files.
- Integrity invalidation: disk reads verify per-chunk SHA-256 (`sessions/lanes.zig` `Lanes.work`) and
  header key/first_page/page equality plus family-name checks (`disk.readFile` → `error.BadHeader`,
  `error.ChecksumMismatch`); restore identity guard `restore.Want{pos,key}` + `index.holds(id,pos,key)`
  (`sessions/restore.zig:23-25`) — the leader names the entry before any rank mutates its slot.
- Nothing in the RAM prefix key includes layout, dims, checkpoint, or model: two models sharing one
  `Store` would hit each other's entries on identical token prefixes.

## 3. Paged KV pool structure and tiers

`sessions/pool.zig` (host book only — bytes live behind a vtable):
- `Family{name, ratio, row_bytes, split}`; `Layout{families, page=256 tokens}` (power of two,
  `page % ratio == 0`). dsv41 families (`families/deepseek_v41/kv/layout.zig`): per KV-source layer two
  families — `comp.L` (compressed KV, fp8 584 B / fp4 384 B a row) and `index_k.L` (fp8 132 B / bf16
  256 B); release topology = 4 KV-source layers (2, 8, 14, 20) → 8 families, 1,790 B/token replicated,
  1,060 B/token with split TP=2.
- `Pool.init(gpa, layout, tokens, world, rank)`: `npages = tokens/page`; TP `world ≤ 8` shards *split*
  families by `logical page % world` (residue mod W free sets `free[W]`, `takeAt(k)` keeps residue,
  `owns/localPage/familyPage` + `discard` page for other ranks' rows). Per-rank device tensor sizing:
  `familyTensorPages` = npages+1 (null page) or local_pages+2 (local null + discard).
- `Slot` = page table `logical i → physical page` (`pages[]`, `len`), admission `quota`/`over` (reserved
  per residue: `span()`+`reserved[]`, `available()` = W × scarcest residue minus reserved), `dirty_lo/hi`
  for device table upload (`takeDirty`), `origin {key,pages}` = the disk file its leading pages came from
  (enables delta parks). Ops: `ensure(end)`, `adopt(shared)` (RAM resume: share full pages, residue
  checked), `push(pg)` (copy-on-write of a partial page), `truncate(keep)`, `releaseAll`.
- `freeset.zig`: `FreeSet` = 6-level bitmap (a bit a page → a bit a non-empty word…), lowest-first
  `takeLowest` in O(log64 n).
- Quota math: `Pool.needPages(prompt, max_tokens, capacity, slack)`; dsv41 admission
  (`families/deepseek_v41/kv/sess.zig:14-29`) computes `need`/`want` (a RAM hit's full pages are shared,
  not new) and asks the store for leader-ordered `spills` (`wait: bool` if even a full eviction won't fit).

Bytes tiers (what a `PageStore` vtable sits on — `sessions/pagestore.zig`):
- **HBM**: `families/deepseek_v41/kv/device.zig` — one CUDA device tensor per family, stacked
  `[slots, pts] int32` page tables per slot; read/write/copyPage via async copies; `fence/waitFence/
  publish` rendezvous with the compute stream. dsv41 wires the store onto this (`sessions_gpu.zig:198`
  `Store.init(..., kx.dev.store(), ...)`), so "RAM tier" here means *pool-resident (HBM) pages held by
  entries*; the bounded state (SWA rings 43×256×584, ratio-2 carries, Engram lookback, DSpark taps —
  `kv/budget.zig:9` `slot_bytes`, `kv/sess.zig` `Measure`) is snapshotted host-side and rides as blobs.
- **Host RAM**: `sessions/pagestore.zig` `HostPool` — the same page tensors in host memory (CPU/test
  engines; per-family tensors sized `familyTensorPages × pageBytes`). `store_residency_*.zig` exercise the
  residency budgets over the real registry.
- **NVMe**: `sessions/disk.zig` `Disk` — files `<root>/<compat[0..8] hex>/rank<r>/<key-hex>.tfs`,
  `O_DIRECT` (buffered fallback + `fadvise DONTNEED`), `.tmp`+`rename` atomic publish, format
  `sessions/format.zig` (`TFSESS02`: Head{key, base, compat, tag, kind, page, world, rank, first_page,
  npages, pos} + segments `ids`/`pool.<family>`/`blob`, each chunked to 4 KiB spans with per-chunk
  SHA-256 hashed by `sessions/lanes.zig` lane threads through bounded `Staging`). **Delta parks**: a new
  turn writes only pages past `origin` stacking on its `base` file up to `max_chain` (8);
  `Disk.extents` walks base-first for restores. LRU `trim` over `budget` (default 64 GiB); the charge is
  physical bytes, or *logical* (all ranks' bytes/world for split families) per `sessions/disk_charge.zig`.
  `scan()` at restart drops `.tmp`, foreign (compat/rank/world mismatch → `error.Foreign`), truncated and
  checksum-failed files, and re-registers bases before deltas; `Store.reconcile` rebuilds the prefix trie
  via `insertFrom`. `sessions/disk_namespace.zig.prepare` takes an exclusive `flock` on the
  compat/rank dir and (with `fresh_start`, which dsv41 passes) purges it at boot.

## 4. Startup memory budgeting

Three cooperating layers; the `PromptCachePlan`/`PromptCacheCapacity` in
`core/engine_cache_plan.zig` are pure metadata carried by `engine_api.zig` (`Info.prompt_cache`,
`Info.prompt_cache_capacity`) into `/metrics` (`server/metrics.zig`) and log lines.

1. **Device budget** (`native/cuda_memory.zig`): `Pool{free,total,reserve,limit,unified}`,
   `room(held) = free − reserve` (capped by `TENSORFOLD_CUDA_MEMORY_LIMIT_GB`). Reserve =
   `TENSORFOLD_MEMORY_RESERVE_GIB` or default `max(at_least, total/floor.share)` — discrete GPU: share 32,
   floor 1 GiB; unified: share 10, floor 4 GiB (`floor()`/`reserveBytes`, lines 138-154).
   **Honest note: there is no literal "9 GiB free" constant anywhere in the tree** (grep over
   `zig/src` and `docs/` finds none). It corresponds to this default share at realistic card sizes
   (total/32 ≈ 9 GiB on a 288 GiB part); the CUDA load-failure message in `native/cuda.zig:279` prints the
   "{free} GiB free less a {reserve} GiB reserve" line this produces. Streams fit via
   `budget.admit(room, stream_bytes, free_streams, asked, fixed)` (line 172): `room/stream_bytes` streams.
2. **Prompt-cache fit** (`native/cache_fit.zig`, host/Mac path; `budget.cacheBytes` for CUDA):
   ceiling = 70 % of RAM (`SHARE_PERCENT`) or the load limit, minus `MARGIN = 2 GiB`
   (`platform/memory_policy.zig`) kept for growing prompt buffers; automatic budget = half the remaining
   room with `DEFAULT_MINIMUM = 16 GiB` floor; explicit `--prompt-cache-gib` refused past the room
   (`error.CacheOverCap`) unless `--prompt-cache-over-cap`. `native/cuda_kv_pool.zig` sequences the room:
   engine's own window first, other streams' fixed parts + a page each, then the cache (`kv_pool.plan`,
   `core/kv_pool.zig:125`; `Budget{room, engine, lanes, fixed, cache, snaps, window, kept_ppm, asked}`);
   `warnCapacity` (`cuda_kv_pool.zig:448-462`) says out loud when `budget_bytes < minimum_state_bytes`
   (one kept state can't fit → "lower --context or --parallel, or TENSORFOLD_MEMORY_RESERVE_GIB").
3. **dsv41 sessions pool** (`families/deepseek_v41/kv/budget.zig`, `cuda_kv.zig:49-97`): pool tokens come
   from `TF_DSV41_POOL_TOKENS`; `Boot.fixed = weights + slot rings (slots × slot_bytes) + workspace +
   runtime + session_ram + staging + unions` GiB; `serveCheck` at boot demands
   `MemAvailable − pool − growth − dip ≥ hard_gib` (hard floor 4 GiB, target 5 GiB; measured terms
   growth 1.9/1.7 GiB, dip 4.9/4.2 rank0/rank1, `zig_dip_gib = 3.0`; overrides
   `TF_DSV41_{POST_BOOT,SERVE_DIP,FLOOR_HARD,FLOOR}_GIB`). `afterBoot` re-measures and refuses
   (`error.PoolDoesNotFit`) printing the largest fitting `TF_DSV41_POOL_TOKENS`. Runtime admission
   (`budget.admit`, line 114): every rank's `usable` must cover `ask.price + restore + outstanding +
   hard floor`, and pool pages by residue — verdict `admit | wait_pages | wait_memory`, with the prefill
   transient priced by `prefillPrice`.

## 5. Re-routing one conversation id to a different model — what breaks

**Is KV keyed/namespaced per model? No — not in RAM.** The model identity enters exactly three places:
- `compat = SHA256(family ratio/row_bytes/split × N, world, rank, page, checkpoint[32], selected layers,
  bounded-piece shapes, semantics{config, options, mode, ds_blocks})` —
  `sessions_gpu.zig:183-195` + `session_identity.zig` (`bind` domain string
  `"TF-DSV41-CUDA-SESSION-v2-fp8-584-132"`). It names the **NVMe directory** and is checked per file
  (`Head.compat/rank/world` → `error.Foreign` in `disk.found`).
- The Imprint (learned prompt-cache states) directory `root/<identity:16 hex>`
  (`core/prompt_imprint.zig:26-31`) — family-namespaced on disk.
- The generic prompt cache's host-sizing (`cache_fit`) — accounting only.

Consequences for a MoM fork:

1. **A shared `Store`/`Pool` across models is unsafe.** `prefix.entryKey` hashes only
   `(tag, full-page token digests, pos, tail)` — identical token prefixes under the same tag collide across
   models with different layers/heads/dims. `restoreRam` would `adopt` pages whose bytes are the *wrong*
   model's rows: `Pool`/`Slot` don't know families at all (page ids index *per-family tensors* chosen by
   the caller); the `PageStore` vtable's `fam` index is interpreted against whichever `DevicePool` is
   wired — a second model's row_bytes/ratio/split would silently mis-read, not fault. The disk restore
   path *does* check (`s.family >= sink.families.len`, family-name equality → `error.BadHeader` in
   `disk.readFile`), the RAM path has no equivalent guard.
2. **Each model needs its own pool instance.** `Pool.init` fixes `layout/npages/page/world` forever and
   `DevicePool.init` pre-allocates every family's device tensor (`familyTensorPages × pageBytes`,
   `pool.tensorBytes()`); there is no re-layout. Two co-resident models = two `Pool`s + two
   `DevicePool`s + two `Store`s (each with its own prefix index, LRU, jobs runner). Nothing forbids that
   (all instance state), but nothing arbitrates between them either: the free-set math
   (`available()`, `planSpill`) and dsv41's `serveCheck`/`afterBoot` budget model assume exactly one pool
   per rank, and `cache_fit.footprint()` is process-wide — MoM must partition the device budget *at plan
   time* (sum the models' `Boot.fixed` + pools, one shared reserve) or the second model will blow past
   `free − reserve` at load (`native/cuda.zig:279` error).
3. **Re-routing turns is KV-losing, not KV-corrupting, in today's shape.** Since model B never shares
   model A's Store, the hit rate for a re-routed turn is zero: B re-prefills from token 0 (or from its own
   entries under the same prefix, which is the correct semantics). Cross-model KV transfer would need a new
   format — the bytes are per-family, per-`ratio`, per-`row_bytes` opaque rows; `sessions/format.zig`
   segments and the compat digest make that deliberate.
4. **The `tag` u32 is the natural model-namespace hook** (it already partitions replay vs fast entries, and
   flows through `chain`/`root`, `entryKey`, `Head.tag`, `Rec.tag`), but it is family-interpreted and only
   32 bits; a fork should fold model id into `tag` *and* keep per-model Store instances, since tag alone
   still doesn't guard `HostPool`/`DevicePool` tensor geometry.
5. **Race guard worth keeping on re-route**: `restore.zig` `Want{pos,key}` + `Index.holds` — a slot must
   verify the entry id still names the same logical entry before adopting pages (leader-ordered across TP
   ranks via `graphs.Agreement`, `sessions_gpu.zig:109-110, 200`).

**Global state identified** (process-wide mutable, in/around these subsystems):
- `native/cache_fit.zig:62` `var host_peak: std.atomic.Value(u64)` — process footprint high-water mark
  feeding `hostMemory`/budget; shared across all engines in the process (fine for one process, but it is a
  shared counter MoM loads will co-tick).
- `core/digest_cache.zig` — cross-*process* shared cache at `$HOME/.cache/tensorfold/file-digests/v1/
  index.bin` (`defaultRoot`, line 87), keyed by (device, inode, size, mtime, ctime) (`digest_key.zig`),
  guarded by `digest_lock.zig` stripe + index file locks. Model-independent (file digests), safe to share.
- Per-instance but threaded: `sessions/jobs.zig` `Runner` tier thread (one per `Store` with a disk),
  `sessions/lanes.zig` `Lanes` pool threads + `Staging` (one per `Disk`), each with mutex/cond — so two
  models = two lane pools (tune `TF_DSV41_SESSION_STAGING_MIB`/lanes per model).
- `sessions/disk_namespace.zig.prepare` — advisory exclusive `flock` held on the compat/rank dir for the
  Disk's lifetime: two engine instances with the *same* compat+rank (i.e. the same model twice) cannot
  share a disk root concurrently; different models get different dirs.
- Env-var reads at init (`optionsFromEnv`, budget knobs) — per process, so two models in one process
  share all `TF_DSV41_*` settings; a MoM fork needs per-model knob scoping.
- Everything else in `sessions/` is instance state: no file-level `var` in any `sessions/*.zig`, no
  globals in `pool.zig`/`store.zig`/`prefix.zig`/`disk.zig` (only atomics inside structs, e.g.
  `jobs.Job.state`, `Lanes.hash_ns/io_ns`).

## Minor observations

- `core/cache_modes.zig` is not cache "modes": it's the canonical decoded-row `Span` algebra
  (validate/equal/prefix/reply) the prompt cache stores per entry (`Entry.decode_spans`).
- `core/checkpoint_cache.zig` is page-cache hygiene (`fadvise DONTNEED` sweeps over safetensors after
  load), not a KV structure; `core/digest_cache.zig` is the checkpoint/file SHA-256 cache described above.
- `sessions/sessions.zig` is just the umbrella module/test aggregator.
- Store's own defaults (`store.zig` `Policy`): `ram_bytes = 512 MiB`, disk `min_tokens = 1024`,
  `max_chain = 8`, budget 64 GiB; dsv41 overrides `ram_bytes` to 256 MiB
  (`TF_DSV41_SESSION_RAM_MIB`) and defaults `skip_covered=false, delta=false` (`sessions_gpu.zig:25-41`).
