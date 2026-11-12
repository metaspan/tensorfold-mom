# Mixture-of-Models (MoM) fork — the diary

This branch extends [TensorFold](https://github.com/ashhart/TensorFold) (upstream,
retained here in full as `main`) into **Mixture-of-Models serving**: one process,
one OpenAI endpoint, N models resident on one GPU, request-level routing, and —
the centerpiece — **shared KV between a heavy model and a thin model derived from
it** (the "nested upcycle").

Everything below was built and measured on a DGX Spark (GB10, sm_121, 128 GB
unified memory). This directory is the full working diary: the design brief, the
per-component exploration notes, the numbered design docs, the raw build/test logs,
and `PROGRESS.md` — the day-by-day log with decisions, dead ends, and the failures
that shaped the design. Where a claim is made, it cites a commit, a probe script
under `experiments/`, or a log in `notes/`.

## The journey, in order

| Date | Milestone | Where the evidence lives |
|---|---|---|
| 2026-11-09 | Brief + subsystem exploration (engine, server, sessions, web layer) | `docs/00`, `docs/01`, `notes/explore-*.md` |
| 2026-11-10 | **Track A** (multiplexed native children) — Gate A PASS: trivial lane 0.11 s e2e under 6-way heavy-lane saturation, router p99 75 µs, 0/300 leaks, strict token equality 8/8 per lane | `experiments/results/RESULTS.md`, `docs/02`, `docs/03` |
| 2026-11-10 | **Track B** B1: unmodified-architecture native fork boots **both lane engines in one process** on the GB10, one endpoint, memory reserve charged once | `docs/06`, `notes/b1-*.log` |
| 2026-11-10 | B2 per-engine routing + explicit model ids; B3 KV isolation proofs; B5 request-start state store (85 % cold / 99.5 % warm prefill reuse, trivial TTFT 46→11 ms); B7 stream priority measured inert and retired | `docs/07`, `docs/08`, `docs/09`, `docs/11` |
| 2026-11-11 | **S1 gate, negative result**: ridge-fit linear cross-model KV translation between independently trained checkpoints measured NOT VIABLE (cross-corpus top-1 0.3418 vs a pre-registered 0.99 gate) — plus the memorization lesson that killed a false same-conversation GO | `docs/12`, `experiments/s1_align.py`, `experiments/s1_fit.py` |
| 2026-11-11 | Pivot to the **nested upcycle**: thin student *extracted from* the heavy model's frozen layers — fidelity 1.0 by weight identity, shared cache as a page-table view | `docs/13` (nested-upcycle spec) |
| 2026-11-12 | M1: NVFP4 checkpoint byte-surgery extractor (safetensors header arithmetic, per-layer config-table truncation, `--verify` sha256 gate) | `experiments/nested_extract.py` |
| 2026-11-12 | M2: teacher-forced bit-identity gate — nested frozen-prefix K/V rows bit-identical to the parent's on the same token stream | `experiments/m2_*_probe.py`, `docs/13` |
| 2026-11-12 | M3 CLOSED: adoption of parent KV state at adopt-time on device, 5× byte-identical across reboots, via the **tail-zero policy** (parent bytes copied only inside the frozen prefix; head-duplicated tail regions cold-start) — commit `c179d63` | `experiments/m3_adoption_probe.py`, `docs/13` |
| 2026-11-12 | Distillation pipeline: nested materializer (found + fixed an fp8 `weight_scale` bug — 3×10³ inflation), streamed per-layer saves, tail-only SFT (loss 4.70→1.14; smoke corpus too small for coherence) | `experiments/nested_materialize.py`, `experiments/distill_train.py` |
| 2026-11-12 | **M4 SharedView**: parent pins its whole KV pages in the shared pool; the thin engine skips their copy, injects the parent's physical page ids into its device page table, and its kernels resolve reads into the parent's arena — writes never leave the thin cache; any gate miss falls back to the copy path. Host gates green (`test-kv-pool`), device gate in flight | `docs/14`, `zig/src/families/qwen3_5/cuda_{cache,pages,reuse,native}.zig`, commits `cbfb3ec` + `9b1ebc4` |

## What is true today (no marketing)

- Track A is deployable and Gate A-verified. Track B runs both engines in one
  process with routing, isolation and request-start state proven.
- Cross-model KV translation for *independently trained* models is a measured
  dead end (S1) — the design docs say so loudly, because the false-positive path
  (same-conversation holdout → memorization) is the most instructive artifact here.
- The nested program works at every gate run so far: byte-exact extraction,
  bit-identical teacher-forced K/V, deterministic adoption with byte-identical
  replies. The shared *view* (zero-copy read of parent pages) is wired and
  host-verified; its device gate is the live work.
- Pre-distillation the thin model outputs gibberish *by design* (frozen 16-layer
  prefix, cold tail); its coherence is a distillation deliverable, not an
  adoption-gate criterion.

## Index

- `PROGRESS.md` — the day-by-day diary (decisions, burned windows, pitfalls).
- `docs/00..05` — brief, findings, component options, implementation & test plans, model selection.
- `docs/06..11` — Track B components (B1 boot, B2 integration, sleep/wake, cache effectiveness, duplication ceiling, stream priority).
- `docs/12` — shared-KV plan + the S1 negative result.
- `docs/13` — nested-upcycle work spec (extraction, GDN head duplication 16v→32v, distillation recipe).
- `docs/14` — M4 SharedView design (page-table shared-cache view in `cuda_cache.zig`).
- `notes/` — exploration maps and raw build/test logs from the B-series baselines.

## Attribution

- Engine + upstream: [ashhart/TensorFold](https://github.com/ashhart/TensorFold)
  (this fork's `main` mirrors upstream; our work is on `mom/main`).
- kessel (Schneewolf Labs, MIT): surveyed for FP8/FP4-aware checkpoint surgery;
  several checkpoint-packaging design ideas (overlay/delta checkpoints,
  requant-stats auditing, promote-changed-to-bf16) were adopted **as designs**.
- mergekit (Arcee AI, LGPL-3.0): surveyed; not usable on NVFP4 checkpoints;
  design ideas only (declarative recipes, refuse-by-default policy).
