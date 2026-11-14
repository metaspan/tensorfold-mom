# TensorFold MoM — two-day technical review (2026-11-12 → 2026-11-14)

Scope: nested-upcycle student program — shared-cache adoption (M4/BL-29), routing
performance gates (BL-6/BL-6T), MTP drafter (BL-21 → BL-30), and the student
distillation recipe (BL-12). All numbers below are measured on the DGX Spark
(GB10, 121 GB unified RAM) unless marked as reviewer estimates.

---

## 1. What we tried and what worked

### BL-29 — shared-cache adoption (M4 device gate) — RESOLVED
- Adoption (thin student reusing the heavy parent's kept GDN state) emitted
  garbage. The leading theory ("parent save writes only 8 regions") was **killed
  by evidence**: `cache.state` is laid out as PLANES —
  `[rec planes: deltas×2 MiB][conv planes: deltas×48 KiB]` — not interleaved
  `[rec|conv]` regions. Every earlier byte analysis had been reading misaligned
  windows and was declared void.
- Fix: plane-correct sanitizer (`918da4834`): rec planes checked f32-isFinite,
  conv planes checked bf16 exp-all-ones (the old filter also missed `inf`).
- Result: adoption now generates 48 real tokens on the toy pair; the copy path
  itself was exonerated (17/17, single DtoD span, save logs full piece length).
- Root-cause class: stateless `tf_q35_page_base` plus four reader sites
  (earlier M4 work). Host-side page-table mock now covers this class; the
  device-copy leg is still untested (see open items).

### Distill pipeline mechanics — verified sound
- **Export integrity**: t2/t3 students load 279/279 tensors; the frozen prefix
  (layers 0–9 + embed + norm, 176 tensors) stayed **byte-identical** to the
  warm-start twin through every training run — the shared-cache frozen-prefix
  digest invariant survives distillation. This matters: it means a gated
  student can still adopt parent state without re-auditing.
- **Coherence gate** (`distill_gate.py`): 12 QA items, ≥10/12 bar, greedy,
  thinking-off, zero template-leak requirement — pre-registered, never moved.
- Twin control (untrained warm-start) also scored 0/12 → early failures were
  corpus scale, not an export bug.

### Key engine facts learned (and worked around)
- The native CUDA engine **categorically refuses plain bf16 MoE** checkpoints
  (`cuda_geometry.zig:290 → UnsupportedQwenQuantization`). bf16 students are
  therefore gated via the HF stack (`distill_gate --engine hf`, a pre-run
  harness amendment; items/bar unchanged). Native gate stays registered for a
  future NVFP4 student.
- The engine advertises **logprobs: false** on the CUDA qwen3_5 backend → no
  in-engine teacher forcing / logit retrieval. Workaround: compute teacher
  logits offline (below).
- MTP head semantics fully reverse-engineered from `cuda_mtp.zig` /
  `qwen36_mtp.cu`: x = rmsnorm(emb(follow))·W_e + rmsnorm(h_t)·W_h → 1 GQA layer
  (gated attention, `[q_h|gate_h]` **interleaved per head**, partial rotary
  64/256 dims, θ=1e7, per-head q/k norms) → 1 MoE (top-8/256, gate-rows-first
  gate_up, norm_topk enforced) → mtp.norm → shared lm_head. Predicts t+2 from
  the target's last-layer tap.

---

## 2. What didn't work

### Decode ceiling is intrinsic without a drafter
- BL-6 r3: TTFT **PASS** (thin 0.046 s vs stock 0.113 s; bar ≤0.85× stock).
  Decode **FAIL**: 141.9 tok/s vs pre-registered ≥154 → NO-GO for the
  speed-based routing case.
- BL-6T (single pre-registered tuned run, explicit "no config fishing"):
  141.4 tok/s → the ~142 ceiling is **intrinsic**, not a config artifact.
  Thresholds were NOT moved (S1 lesson).

### More same-distribution data does NOT buy coherence
| run | corpus | epochs | eval loss | gate |
|---|---|---|---|---|
| twin (untrained) | — | — | — | 0/12 |
| t1 | 175 rows | 2 | ~2.0 | 0/12 |
| t2 | 2,492 clean rows | 4 | plateau 1.99 | 3/12 |
| t3 | 12,226 clean rows | 3 | 1.88/1.76/1.72, no plateau | **3/12** |

- 4.9× data moved the loss curve but produced **zero gate movement**.
- Failure analysis: t3's misses are **knowledge** failures, not format
  failures — 23+19→"22", 6×7→"60", 10 km→"10.21 mi", 1.5 h→"2.5 minutes",
  "reverse 'world'" → rambles. **Correction 2026-11-14 (review follow-up):** the
  "parent answers all 12 correctly" line was an overclaim from spot-checks during
  the t1 parent-control episode, not a scored 12-item run. A scored teacher-gate
  run on the dequantised bf16 teacher is queued (doubles as KD teacher calibration
  per both reviewers' ask); until it lands, treat teacher correctness on the gate
  items as spot-verified only.
- Conclusion: teacher-rollout sampling on the trivial lane teaches the
  student the teacher's **shape, not its answers**. Shape-learning saturates
  fast (t2 already plateaued on eval loss); correctness needs a signal that
  constrains the tail's output distribution token-by-token → logit-KD.
- This is the spec's original recipe ("distillation incl. thinking-blocks")
  catching up with us: pure sampled-CE distillation is a weaker surrogate.

### MTP head can't be borrowed zero-shot
- Parent's 19 `mtp.*` tensors copied verbatim beside the thin checkpoint
  (`mtp_merge.py`) — engine serves it (`drafts: MTP`), but acceptance was
  **1/40 (~2.5%)** and decode **130.9 tok/s — worse than stock 141.9**.
  The head must be trained against the student's own hidden states.
- Trainer (`mtp_train.py`, 257 lines) is validated through step-1
  (loss 10.703, sane) but the full smoke + export leg has never completed
  (killed by OOM contention beside the collector). Queued solo.

### Stack landmines hit (all documented, most pinned)
1. **Config-form saga**: bare `Qwen3_5MoeForCausalLM` config → native no-boot;
   parent VL-form copy → `MissingTensor embed_tokens`; bf16-variant copy →
   quantization refusal. Resolution: the trainer now ships
   `model.config.save_pretrained` (whatever form it loaded in); do not hand-swap
   configs on students.
2. **safetensors gotchas** in the merge tool: `data_offsets` are
   header-relative (start at 0, not 8); single-file `model.safetensors` is
   missed by `model-*.safetensors` globs; `.jinja` templates easy to drop.
3. **Dependency hell on the KD path**: FP8 linear needs the `kernels` package;
   `pip install -U kernels` dragged huggingface-hub to 2.4.0, breaking
   tokenizers/datasets pins (transformers 5.8.1 requires hub ≥1.5,<2.0);
   kernels 0.18 crashes transformers' hub-kernels import ("Either a revision or
   a version must be specified"); kernels ≤0.11 force hub <1.0. **Working pin:
   kernels==0.12.3 + hub 1.x.**
4. **FP8 triton kernel unusable here**: the hub's finegrained-fp8 kernel dies
   with triton_kernels "Unknown recipe" on this stack → fallback:
   `dequant_fp8_teacher.py` produced a plain-bf16 67 GB teacher copy (e4m3 +
   128×128-block bf16 scales, transformers finegrained-fp8 semantics).
5. **Unified-RAM wall (open)**: on GB10 host RAM = GPU RAM (121 GB). Loading
   the 67 GB teacher with `device_map="cuda:0"` gets OOM-killed at ~97% of
   weight materialization: pinned GPU weights (67 GB) + mmap'd safetensors page
   cache (67 GB) are charged to the same cgroup, and mmap pages stay pinned
   because transformers holds **all shard handles open** and materializes
   lazily via `get_slice`. Attempts that failed: `.to(dev)` cycling (double
   counting), device_map direct (same), `posix_fadvise(DONTNEED)` on
   `__exit__` (never fires — handles close only at the end), read-tracking
   eviction (never fires — the loader bypasses `get_tensor`). No sudo → no
   `drop_caches`. **Discovered fix, pending run**: the loader honors
   `disable_mmap` (reads each shard fully into anon memory instead of mmap;
   ~1.7 GB transient per shard, no pinned page cache) and `from_pretrained`
   takes the kwarg.

---

## 3. Decisions taken (with rationale)

- **BL-21 → route (b)** (user): build a trained MTP drafter head for the
  nested checkpoint, rather than accepting TTFT-only routing or dropping the
  speed case. Justification: decode shortfall is intrinsic (BL-6T), and the
  zero-shot-control experiment proves the head — not the plumbing — is the
  missing piece.
- **Dependency chain**: MTP head trains only on a GATED student (≥10/12). If
  the tail retrains, the head retrains. Hence BL-30 blocks behind the
  distillation gate.
- **Logit-KD over more data** (user, after t3): 5× data bought zero gate
  movement → fifth-scale-up has a bad prior; the token-level signal is the
  principled lever. Rejected-for-now alternatives: 50k-row rescale,
  thinking-mix-only tuning, stop-and-write-up.
- **Pre-registration discipline** (S1 lesson, held all two days): gate bar
  ≥10/12 and decode bar ≥154 never moved; KD hyperparameters (β=0.5, T=1.0,
  top-k=8) fixed in the BACKLOG *before* the run; BL-6T ran exactly once.

---

## 4. Options left open (deliberately)

- **Native-mode gate revival** for an NVFP4 student (needs a quantizer path:
  mergekit's HF loader dequantizes/refuses block-quantized; `nested_extract.py`
  remains the only thing that lays bytes on our NVFP4 checkpoints).
- **Engine logprobs** for qwen3_5 CUDA (currently advertised false) — would
  enable on-policy / student-rollout KD and engine-side logit arbitration.
  Offline top-k KD is the workaround; engine work is the durable fix.
- **BL-28**: verify the MoM park actually drops the heavy lane's kept states
  (prompt-cache/park claim, reviewer-unconfirmed).
- **BL-22 residuals**: AdoptMap/evict/drop coverage, point-of-use checksums,
  device-copy-leg tests.
- **Production SharedView pair** never run end-to-end green post-BL-29
  (toy green only).
- **Concierge UX** (thin model as task-ledger front-end) — parked by user.
- **`:8000` restore** — held by user.
- Upstream PR packaging: `docs-mom/UPSTREAM-PRs.md`, strict mirror cron;
  no-GPU CI red inherited from upstream, documented not chased.
- kessel (Schneewolf-Labs) stays separate; harvest packaging ideas only.

## 5. Planned approach to the remaining challenges

1. **Finish the KD data path** (tonight): run prep with `disable_mmap=True` →
   teacher loads inside the cgroup; validate the 4-row dry-run (shapes + teacher
   top-1 vs corpus next-token agreement — the calibration gate for the whole
   KD premise). Keep the fadvise wrapper as belt-and-braces.
2. **Full prep**: 12,226 rows → `kd-10k.pt` (~2–3 h, top-8 per position,
   ~300–400 MB).
3. **t4**: same corpus, loss = 0.5·CE + 0.5·T²·KL(top-8), 3 epochs, frozen
   176/176 verify, gate t4 (unchanged bar). Success = ≥10/12.
4. **If t4 passes**: solo MTP trainer smoke (export leg!), train head on gated
   t4, `mtp_merge`, serve, measure decode vs ≥154 with the MTP drafter →
   BL-6 GO/NO-GO evidence for the routing case; then production SharedView
   end-to-end.
5. **If t4 plateaus again** (pre-committed escalation ladder, each step
   pre-registered, no config fishing):
   a. KD shape check first: is the KL term even alive (loss decomposition,
      KL-per-epoch curve; β sweep as a declared 3-point matrix)?
   b. Knowledge-targeted corpus: seed generator weighted toward the miss
      categories (arithmetic, unit conversion, comparison, spelling ops) —
      still disjoint from the 12 gate items.
   c. 50k-row scale-up (only after b, since pure scale was already refuted
      at fixed distribution).
   d. Engine logprobs request (on-policy KD) as the durable fix.

## 6. Ask for reviewers

- Is top-k=8 sufficient for the KL on a 152k-vocab MoE teacher? (Options
  considered: k=32 costs ~4× storage; tail-mass term currently renormalized
  away.)
- Any objection to fp8-dequantized-bf16 teacher vs native NVFP4 teacher as
  the KD target? (Same base weights; quantization noise ~1 recipe away.)
- Sanity-check the t3 failure taxonomy — if you can reproduce a miss category
  that is actually a *format* failure, the corpus-diversity branch (5b) gets
  heavier weight.
- BL-6 decode bar: ≥154 tok/s pre-registered; MTP acceptance ~85%+ is
  required to clear it from 141.9 base. What acceptance do you consider
  realistic for a trained 1-layer MTP on a 0.8B-ish tail?
