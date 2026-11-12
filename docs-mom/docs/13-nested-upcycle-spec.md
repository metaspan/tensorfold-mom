# 13. Nested upcycle: shared KV by weight identity (work spec)

Status: PROPOSED. Follows docs/12 run 3 (translation NO-GO: cross-corpus
top-1 0.3418 vs 0.99 gate). Prize: docs/10's cold-prefill duplication.
Grounded against the real 35B checkpoint tensors and the fork's
`cuda_cache.zig` / `cuda_reuse.zig` / `cuda_native.zig` as of `6d082a5`.

## 0. The construction principle (and a correction)

Sharing KV rows requires MORE than identical K/V projection weights: a row at
layer L is a function of the residual stream entering L, i.e. of ALL layers
below it. Extracting the ten full-attention layers in isolation (my earlier
sketch) does NOT yield shareable rows — an independent student's residual
streams differ and the rows differ. The only exact construction:

**Thin = a frozen PREFIX of the heavy stack + a trainable tail.**
thin = heavy.layers[0..N] (bit-identical bytes) ++ student tail layers.

Consequences that make this BETTER than the attention-only sketch:
- Rows for layers 0..N — attention K/V *and* GDN recurrent/conv state — are
  bit-identical to the heavy model's by construction. GDN states, which
  translation could never touch (16v vs 32v shapes, recurrence), come along.
- Bidirectional adoption: thin gets the prefix free after heavy prefills;
  heavy can resume its layers N+1..39 from thin's layer-N residual after thin
  prefills a prefix. Prefill work splits across lanes.
- The student's tail reads exactly the residual stream the heavy model reads —
  distillation trains the tail on the true frozen-prefix representation.

## 1. Frozen-prefix extraction (M0, device-free)

Grounded inventory (safetensors headers, `nvidia/Qwen3.6-35B-A3B-NVFP4`
@1355db6a):
- per layer ~465 MiB (experts 432 MiB = 3072 tensors; attention stack ~28 MiB;
  GDN stack fp8 in_proj_qkv 8192x2048, in_proj_z 4096x2048, out_proj
  2048x4096, conv1d 8192x1x4 bf16, A_log/dt_bias, in_proj_a/b, norm 128)
- attention (at 3,7,...,39): q_proj 8192x2048 fp8 + q_norm 256 bf16,
  k/v_proj 512x2048 fp8 + k_norm, o_proj 2048x4096 fp8. QK-norm INSIDE the
  projection block — shared rows inherit it identically.
- embed_tokens 248320x2048 BF16 (~970 MiB), lm_head fp8 (~254 MiB+scales)
- layers 0..9 total **4657 MiB**; full 40 layers 18617 MiB

Extraction script `experiments/nested_extract.py` (torch-free: safetensors
headers + raw byte slice):
1. Copy verbatim: `model.language_model.embed_tokens.weight`,
   layers 0..N (ALL their tensors incl. fp8 weights + every scale tensor —
   NEVER requantize; byte copy is the correctness mechanism),
   `model.language_model.norm.*`, heavy `lm_head.*` (frozen default;
   trainable option decided by eval).
2. Append tail layers N+1..N+T as TRAINABLE tensors (section 2).
3. Emit `config.json`: same family (`qwen3_5_moe`), `num_hidden_layers =
   10+T`, `full_attention_interval=4` (tail keeps the 3-mod-4 pattern so the
   loader sees only known layer kinds), frozen prefix hyperparams untouched,
   new field `nested_frozen_layers: 10` + `nested_parent: <35B sha>`.
4. Emit `NESTED.json`: `frozen_prefix_digest` = sha256 over (sorted frozen
   tensor name+bytes), parent checkpoint sha, geometry triple
   (kv_heads=2, head_dim=256, page layout, kv8 flag), vocab.json digest,
   chat template digest. This digest is the adoption permission (section 4).
5. Host-testable: digest reproducibility, tensor completeness vs parent
   index, no requantized dtype drift.

Tail architecture (student-only, unconstrained by the heavy checkpoint):
GDN+attention on the family pattern, FFN = dense or shared-expert-only
(1 shared expert of the A3B MoE) — tail layers ~30 MiB each vs 465 MiB heavy.
Default pilot: N=9 (10 frozen), T=6, tail FFN shared-expert-only dense.
Thin active compute ~ (10/40 x 3B) + tail ~ 1B; thin extra RAM = tail only
(~200 MiB) once weight sharing (below) is in; without it, +4.66 GiB.

**Weight sharing by reference** (same process, GB10 unified memory):
`openLanes`/`Engine.init` load weights per-open from a dir; add a process-wide
weight registry keyed by (parent sha, tensor name): the thin engine's frozen
tensors resolve to the heavy engine's live `DeviceBuffer` — the 4.66 GiB
prefix + 970 MiB embedding + 254 MiB head become zero extra cost and make
row identity a tautology (same bytes feed both kernels).

## 2. GDN heads: what 16v->32v duplication IS and IS NOT

- NOT for the prefix. Frozen GDN layers keep the heavy's exact shapes
  (in_proj_qkv 8192 (=16k x128, 32v x128), conv1d 8192x1x4, ...). Their
  recurrent/conv state is SHARED by being the heavy engine's own state —
  duplication is not involved and faking 16v state into 32v slots would be
  the same unfaithful-translation error S1 just measured dead.
- FOR the tail (optional compute optimization): student tail GDN layers may
  train at 16 value heads and deploy at 32 by head-duplication ONLY IF the
  32v weights are CONSTRUCTED by duplicating the trained 16v halves (pair-
  equal by construction: same conv1d channel groups, same out_proj column
  pairs). Then the 32v state is a free view of the 16v state and any future
  heavy-shaped consumer can read it. Halves that merely "happen to be
  similar" are invalid — enforce by weight tying in training + a structural
  equality assertion in the export.
- The heavy model's 32v states are never rewritten; thin only ever writes
  thin-owned tail state.

## 3. Student-only distillation recipe

Trainables: tail layers (incl. their qk-norms, GDN A_log/dt_bias/convs),
`model.language_model.norm` (final norm), optionally lm_head. NEVER touched:
embed_tokens, layers 0..9, anything covered by `frozen_prefix_digest`.
The trainer physically cannot drift frozen bytes: it loads them read-only and
export re-verifies the digest byte-for-byte (same tool as M0 step 4).

- Teacher: the heavy 35B engine itself; logits from the TensorFold engine so
  distillation sees deployment NVFP4 numerics (no bf16->fp8 train/deploy skew).
- Data mix:
  1. on-policy student rollouts on the trivial-traffic distribution
     (router logs after B5), sequence-level KD (KL over teacher full-vocab
     distribution, T=1) + hard-token CE;
  2. heavy-lane transcripts WITH thinking blocks in history — sticky
     conversations hop lanes; the tail must condition on `thinking` content
     it never generates (answering Derek's question: template unification is
     config, thinking-comprehension is THIS data line);
  3. prefixes prefilled by the heavy engine through turn boundaries, so the
     tail trains on the true frozen-prefix residual stream;
  4. anti-forgetting: B5 + corpus B replays (byte-identity fixtures exist).
- Precision: bf16 master weights for the tail; engine consumes exported
  fp8/quantized tail tensors — export gate = tail-only KL drift < eps vs
  bf16 tail.
- Gates (all cheap, harness already built):
  - G1 identity (device): with both engines serving a shared prompt,
    TF_MOM_KVDUMP rows for layers 0..9 + GDN state must be BIT-identical
    across engines. Expected 1.0 by construction; any miss = loader/digest
    bug. This is the exact-decode guarantee's regression test.
  - G2 quality: thin >= stock Qwen3.5-2B win-rate on trivial traffic
    (paired judge or exact-match set); consistency-with-heavy on misroute
    replays (misroutes get near-agreeing answers, cheaper than today).
  - G3 adoption: heavy prefills -> thin adopts (section 4 copy path) ->
    decode-resume token stream byte-identical to native thin prefill
    (B5 machinery, both corpora).

## 4. Shared-cache view in `cuda_cache.zig` (grounded)

Facts from the code:
- `Paged` = { base: u64 (page 0 of the CORE POOL arena), table: device int
  page table, ids: host u32 slice (pool lane maps its lease's ids), tokens };
  `elems()` sizes a page as EVERY attention layer's rows for `tokens` —
  page layout is per-model (10 layers/page heavy vs 2/page thin).
- Recurrent/log/before via `recurrent.Pool = Workspace(DeviceBuffer)`:
  slots are BINARY leases (`leased: bool` + generation), not refcounts;
  `collect()` recycles unleased resources — cannot be shared as-is.
- Adoption today exists per-model via `cuda_reuse` kept-state copies
  (the very bytes kvdump reads; layout documented in `kvdump_format.zig`:
  per dumped layer K then V, per head, per position; fp8 rows + block scales).

**v1 — copy-based adoption (no kernel surgery), M3:**
heavy's `cuda_reuse.save` checkpoint (layers 0..N slices) is published into a
process-wide `NestedStore` keyed by (frozen_prefix_digest, token-prefix hash).
Thin adopts at route time: strided DtoD copy of the 0..N sub-rows from the
kept-state copy into thin's own pages + GDN state slice into thin's leases,
then resume from checkpoint (same machinery B5 exercises). GB10 unified
memory makes this a memcpy-rate cost vs ~0.75B active prefill saved. Refuse
adoption (cold prefill fallback) unless digests match exactly:
frozen_prefix_digest + geometry triple + kv8 + vocab digest + template digest
(docs/02 compat-digest extension point; `context_owner` guards teardown).

**v2 — zero-copy page view (the page-table vision), M4:**
1. Per-layer page layout in `qwen35.cuh` kv_row (or geometry `layer_view`
   stride so thin reads its 2 attention slots as a SUBSET stride of heavy's
   10-layer pages) — kernel + `Paged.elems` change, opt-in behind a flag.
2. `Workspace` entries gain `refcount: u32` (acquire/view/markUsed/release/
   reclaimable/collect semantics preserved; reclaim only when refcount==0);
   shared pages leased by both engines' caches.
3. Thin's `Paged.base` points at the shared arena; ids mapped through the
   existing pool-lane `own_ids=false` path.
4. Eviction order: shared arena owned by the heavy lane; thin's hold =
   refcount; thin's own tail pages evict independently.
Failure modes written down: fp8 scale interleaving must match on both sides
(kv8 layout identical or refuse); conv tails (last 3 tokens) recompute on
resume (B5 does this today); same-process only — device pointers, no
cross-process ambitions; token streams must be template-unified (config,
proven in docs/12 run 1) or the prefix hash lies.

## 5. Milestones and honest economics

- M0 extraction + digest tooling + host tests — **DONE** (`nested_extract.py`,
  49669 tensors, 10.07 GiB, 31034 frozen byte-verified, digest `abd07d04...`).
- M1 native qualification: engine opens nested checkpoint. **DONE** — the
  blocker was config, not loader: the checkpoint is VL-form and the engine
  reads layer facts from `text_config` (num_hidden_layers, layer_types,
  quantized_layers); a top-level `num_hidden_layers` is silently ignored.
  After truncating the tower fields (+ keeping `mtp.*`), the engine opens,
  advertises and answers. Raw truncation gibberish is the expected
  pre-distillation quality floor, not an M1 failure — coherence is G2's gate.
- M2 G1 identity gate on device (kvdump bit-identity, both engines).
  **DONE — PASS** (teacher-forced, `m2_tf_probe.py` + `m2_tf_score.py`):
  9 checkpoints, 8-turn conversation, token streams equal, K+V rows at
  layers 3/7 bit-identical across the two engines at every position.
  Self-replay variant is NOT valid for a degenerate lane (history pollution
  masquerades as token mismatch) — teacher-forcing with the parent's real
  replies is the working recipe.
- M3 v1 adoption + G3 decode-resume byte-identity + B5-style cold-pass A/B.
  **API landed** (`cuda_reuse.zig`: `AdoptMap.buildParams/build`, `adoptable`,
  `adoptCache`; 2 host tests in test-mom-host). Scope insight: while the tail
  is still warm-started parent bytes, the whole 16-layer stack is
  prefix-identical (digest covers 0..9; the tail was byte-copied from the
  parent too) -- so v1 adopts ALL nested layers (state head + kv pieces), and
  the nested engine resumes at `at` exactly like a same-engine reuse. When
  distillation changes the tail bytes, adoption shrinks to the true frozen
  prefix (attention 3,7 + linear states <10) and tail layers must re-run --
  the per-layer-position problem, deferred with post-training handoffs.
  Gate: open-time weight comparison of layers 0..15 across the two loaded
  tables (not blind trust of NESTED.json); `adoptCache` never owns the parent
  copy (parent keeps restoring from it). Paged bf16 goes through the
  pageCopies-shaped 2D path; fp8 paged falls back to per-piece Runs.
  Expected recovery: heavy skips layers 10..39 on thin-seen prefixes ~=
  25% of heavy cold prefill (0.75B/3B active share), thin skips its prefix
  prefill outright; the docs/10 47.3% was the FULL-corpus ceiling incl.
  warm lanes — quote 25% (heavy-side) as the committed number.
  **Trigger surgery map (2026-11-11 code read):** lookups/restores are decided in
  the core lanes (family-neutral); the family only supplies the
  `prompt_cache.Snapshots.VTable`, so a cross-engine trigger cannot live in the
  family alone. Chosen v1 shape: parent-side `AdoptSink` on `reuse.Target`
  (updated inside `reuse.save`, which already sees `s.prompt()` — the kvdump
  precedent) + engine surface `adoptKept()` + nested-side call at the core's
  restore point, gated by TF_MOM_ADOPT=1, mom-paired qwen3_5 engines, and sink
  tokens a prefix of the nested prompt; the lane's own store lookup always wins
  on hit (adoption only covers misses the parent already paid for). Open-time
  gate v1 = NESTED.json digest + B2 vocab-bytes guard + recorded M0 --verify;
  accepted v1 gap: no in-server re-hash at boot (close with chunked on-device
  compare at first adopt).
- M4 v2 zero-copy + refcount pool surgery, device A/B vs v1.
Kill criteria: M1 loader gap that can't be closed without forking the kernel
matrix -> fall back to dense-shape tail; G1 miss -> the byte-copy discipline
is broken somewhere, fix before anything else; G3 miss -> v1 copy path has a
layout bug (kvdump diff is the debugger).
