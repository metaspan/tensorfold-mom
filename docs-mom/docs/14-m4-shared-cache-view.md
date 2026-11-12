# M4: Paged page-table shared-cache view (zero-copy parent → nested KV)

Goal: the nested (16-layer) engine reads the parent's KV pages **in place** — no
adoption copies for the KV pieces — by giving the nested `Cache` a page view whose
physical addresses resolve into the parent cache's arena with a per-attention-layer
row re-stride. The GDN state block stays on the existing adoption path.

## Why this is possible (measured, not assumed)

Page layout (`Paged`, cuda_cache.zig; `kvBytes`, cuda_geometry.zig):

    page_bytes(g) = attentions(g) * 2 * kv_heads * tokens_per_page * row
    row = head_dim * 2 (bf16) or head_dim + 4 (fp8)

Within a page, the per-attention-layer block sits at
`layer_index * cap * row` (per-layer blocks with page-pitch `cap * row`), keys then
values, kv_heads blocks — `Layout.piece` mirrors this for the copy path.

Both engines derive `attentions()` from `layer_types`, in layer order. Nested's 16
layers are the parent's layers 0..15, so nested's attention sequence (3,7,11,15) is
exactly the **prefix of length 4** of the parent's (3,7,...,39). Row content is
byte-equal across engines — **M2 measured it** (teacher-forced K/V bit-identical at
layers 3/7; the M3 byte audit later showed all 15 KV pieces bit-identical).

## Design

### View struct (cuda_cache.zig)

    pub const SharedView = struct {
        parent_base: u64,        // parent cache's paged.base (arena start)
        parent_stride: usize,    // parent paged.tokens * row * 2 * kv_heads ... per LAYER block pitch:
                                 //   parent cache.cap * row, matching piece math
        rows: usize,             // how many leading attention layers the view exposes (4)
        pins: Pin,               // refcount handle on the parent cache's pages (below)
    };

`Cache` gains `shared: ?SharedView = null`. In `seg()` (cuda_cache.zig:290-292),
where `sg.kv = p.base` for the paged case, a shared view instead fills a new
`Seg.ptok`-adjacent field pair (kernel ABI, qwen35.cuh `Seg`):

    sg.kv      = parent_base                      // resolve through the parent arena
    sg.pt_stride = parent_layer_stride            // NEW c_int field (or reuse `ps` slot
                                                  // if it is free for this path — verify
                                                  // in qwen35.cuh before adding a field)

Kernel-side address for (logical page p, row r, layer l, kv h):

    page_bytes_parent(p) = table[ids[p]]          // same page ids (below) or mapped
    addr = sg.kv + l * sg.pt_stride + table_entry(p) + (h_kind * cap_parent + r) * row

with `l < rows` only — nested kernels never index beyond their own `attentions()`.

### Page-id mapping

v1 keeps it simple: the shared view allocates its **own** logical page ids but the
*table entries* are the parent's physical offsets, produced by `AdoptMap` at adoption
time (the sink already records which parent pages covered `[0, at)`). Decoded rows
that exceed `at` fall into nested-owned pages as usual: the view is a **prefix split**,
`shared_pages = at / parent_tokens`, the tail pages are nested-native. No kernel
changes to allocation; `seg()` composes: positions < at resolve through `shared`,
positions >= at through the native table.

### Lifetime (the hard part)

Parent pages are pool-leased (`own_ids = false` → core pool lease). A nested view must
hold a **pin** on each parent physical page it references:

- `Pin` = refcount increment on the core-pool page ids (the pool already refcounts
  for the parent's own lease; add one more holder per view).
- The view registers with the parent target (cuda_reuse `Target`), so eviction or
  `drop()` of the parent copy retires views first (same retire path as the adopt sink).
- If pool refcounting is not exposed where cuda_cache lives, v1 fallback: pin the
  **whole parent lease** (coarse, safe), release on view drop. Measurable waste only
  while the conversation is dead.

### Gate (device, M4 acceptance)

1. Boot parent+nested; parent prefills the hop conversation (sink fills).
2. Nested hop request with `TF_MOM_SHARED_KV=1`: KV pieces resolve through the view
   (log `shared view: n pages of m at stride s`), GDN state still adopted.
3. Nested reply must equal the M3 adoption path's reply byte-for-byte (both are
   garbage pre-G2, but deterministic for the same bytes modulo the ULP non-determinism
   measured on the GDN scan — compare with the GDN state ALSO taken from the same
   adopted copy, which makes both runs start from identical state; KV is byte-stable).
4. `cuMemGetInfo` free-bytes delta proves zero KV allocation for the nested pages
   (the arena grows by tail pages only).

## Surgical site map

| file | change |
|---|---|
| `families/qwen3_5/cuda_cache.zig` | `SharedView` struct; `Cache.shared`; `seg()` resolve; pin/unpin on acquire/free |
| `families/qwen3_5/cuda_reuse.zig` | `AdoptMap` grows `page_offsets`; view built next to `adoptCache`; retire on `drop` |
| `native/qwen35.cu(.h)` | `Seg` extra stride field (or reuse); addressing branch when `kv` marked shared |
| `families/qwen3_5/cuda_native.zig` | lane registers view with parent target; env gate `TF_MOM_SHARED_KV` |

## Open items (blockers to resolve in code, not speculation)

1. `Seg` spare capacity: `ps`/`pt`/`plog` are used by the pending-token path —
   confirm liveness on the paged decode path before stealing one; else extend the
   extern struct in lockstep with qwen35.cuh (both engines rebuild together, ABI
   mismatch is a build-time error).
2. Pool refcount exposure for per-page pins (v1 may pin the parent lease whole).
3. Parent `cap` changes → `parent_layer_stride` must be captured at view-creation
   (cap is fixed per cache in practice; assert it).
4. fp8: `row = head_dim + 4` — same algebra, but verify the scale tail is inside the
   per-layer block (it is, per `Layout.piece` `.fp8` branch).

## Dependency

M4 lands after M3 closes (adoption path is the correctness baseline for the gate).
GDN state stays adopted (copy) — determinism control proved state bytes are ULP
non-reproducible, so state sharing-by-view across engines would re-expose that; the
copy-once-then-own model is right for state, zero-copy is right for KV.
