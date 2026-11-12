# Design Brief (as received, 2026-11-10)

Received verbatim from Derek. My interpretation notes are appended, not interleaved.

## Objective

Fork TensorFold to support a single-engine Mixture-of-Models (MoM) serving path. The
engine loads multiple models, exposes one entry point, manages prefill and KV caches,
and routes each request (or incremental message) to the appropriate model while
preserving full performance on the chosen path. Priority use case: answer trivial
traffic on a permanently hot small model so the larger model stays ready for real work.

## Core Design Principles

- Single process, single OpenAI-compatible endpoint.
- Request-level routing (not token-level MoE).
- Models remain independent checkpoints; no mandatory joint training for the first version.
- Exact decoding guarantees of TensorFold are preserved on every model path.
- Incremental messages may be freely re-routed; session affinity is optional and configurable.

## Component Architecture

1. **Model Registry & Weight Manager** — loads and tracks multiple family packages
   simultaneously. Supports co-residency or fast sleep/wake. Memory budget is global;
   weights can be pinned, paged, or streamed from NVMe.
2. **Request Router** — receives every request before any model forward pass. Decision
   inputs: explicit model field, simple rules (length, keywords, presence of code),
   optional tiny classifier, or load. Emits target model ID and routing metadata. Must
   complete in <5 ms for trivial cases.
3. **Scheduler & Batcher** — per-model queues with a shared continuous-batching loop.
   Trivial-model queue has higher priority. Chunked prefill and in-flight batching
   remain model-local unless a shared-prefill path is later enabled.
4. **KV Cache Manager** — default: independent paged caches per model. Extension points
   for prefix caching within a model and (later) PrefillShare-style or linear-map
   translated caches across compatible models. Hierarchical tiers (HBM / host / NVMe)
   unchanged.
5. **Execution Pipelines** — existing TensorFold family kernels are reused. Each model
   keeps its own lane engine, drafter, and exact verification path. Router simply
   selects which engine receives the request.
6. **Memory & Runtime Orchestrator** — global memory accounting across models. Supports
   pinning the small model, sleeping the large model, and rapid reactivation. Multi-node
   TP/PP remains per-model.
7. **Observability** — routing decisions, per-model TTFT, cache hit rate, memory
   pressure, switch latency. Hooks for the existing dashboard.

## Routing Policy (v1)

- Length < N tokens or matches trivial patterns → small model.
- Explicit model field overrides.
- Otherwise → large model.
- Optional sticky affinity per conversation ID.
- Free re-routing allowed on every incremental message.

## Implementation Order for the Fork

1. Multi-model registry and loader (two models co-resident).
2. Request router with rule-based policy.
3. Per-model queues and scheduler dispatch.
4. Independent KV caches with correct prefix caching per model.
5. Observability and sleep/wake.
6. (Later) shared-prefill or KV translation experiments.

## Small Models to Prove the Hypothesis

| Role | Suggested model | Approx. size | Why |
|---|---|---|---|
| Trivial / always-hot | Qwen2.5-1.5B-Instruct or Gemma-2-2B-it (or 4-bit equivalents) | 1–3 B | Sub-100 ms TTFT, tiny KV, family support |
| Real work | Qwen3-8B or Qwen3.6-35B-A3B (MoE) 4-bit | 8–35 B | Strong for non-trivial; fits with small model on 128 GB unified |
| Optional third | 0.5–1 B draft/specialist (e.g. DFlash2, tiny code model) | <1 B | Tests multi-model routing + speculative paths |

Minimal experiment: load 1.5 B + 8 B together; route messages < 32 tokens (or
"hello"/"status"/"ack") to the small model; measure TTFT and e2e latency on trivial vs
non-trivial traffic, memory headroom and large-model reactivation time, correctness of
independent KV caches when a conversation is re-routed, and router overhead. No
training required. Phase 2 (shared KV / PrefillShare-style / linear-map translation)
out of scope for v1.

## Success Criteria for v1

- Single endpoint serves both models.
- Trivial requests stay on the small model with latency indistinguishable from a
  dedicated small-model server.
- Large model remains available and delivers normal TensorFold performance when selected.
- Incremental messages can be routed to either model without corruption of state.
- Memory accounting prevents OOM when both are resident.

The brief deems itself sufficient to begin the fork at "two-model registry +
rule-based router".

---

# Interpretation notes (mine, informed by source exploration — see 01/02)

1. **"Single engine" vs "router selects which engine"**: the brief's component 5 says
   the router "simply selects which engine receives the request", which admits an
   architecture where N engine *instances* share one process and one HTTP front door.
   Two candidate realizations (native-Zig multi-engine vs multiplexed child processes)
   are compared in `02-component-options.md`; both preserve exactness because each path
   keeps its own unmodified verification pipeline.
2. **Model picks must respect native family support.** Suggested Qwen2.5-1.5B /
   Gemma-2-2B are not necessarily on the native engine's supported family list;
   TensorFold's own small reference model is Qwen3.5-2B (tested on GB10). See
   `05-model-selection.md`.
3. **"Single process" is treated as a target, not a v1 blocker**: v1 validates the
   hypothesis (routing quality, latency isolation, cache independence) on whichever
   path gets there first without touching exactness code; the native single-process
   build is the flagship deliverable of the fork track.
4. **Re-routing correctness definition**: TensorFold KV is model-specific; "free
   re-routing" means the conversation's *text* history moves with the request and the
   new model prefills what it has not cached (prefix cache per model, keyed by tokens +
   model identity). It cannot mean KV transfer. Cross-model KV translation is phase 2.
