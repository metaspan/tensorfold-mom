# B5 — Cache effectiveness over an alternating-lane conversation

**Question.** With the big model kept resident, does a 28-turn conversation that
alternates trivial (→ Qwen3.5-2B) and work (→ Qwen3.6-35B-A3B-NVFP4) turns
(i) route every turn to the right lane under free re-routing, and
(ii) reuse per-lane prefix caches so long histories cost sublinear prefill?

**Method.** `experiments/b5_cache_probe.py`. Single `mom-serve --no-sticky
--parallel 2 --pin` (trivial lane pinned hot, no `--lane-idle` — both engines
resident for the whole run). Greedy decoding (`temperature 0`), thinking off
(`chat_template_kwargs.enable_thinking` — the only form `fields.zig` honors).
Each turn sends the full history (user+assistant) as the prompt; the client
streams so TTFT is time-to-first-content-delta, and asks the server for exact
`usage` token counts (`stream_options.include_usage`). The conversation is
replayed twice: pass 1 cold, pass 2 warm. Per-turn prefill reuse is read from
the engine's own `prompt cache: N tokens, resumed at R, kept K` line, paired to
the turn by exact prompt-token count.

**Finding: the qwen3_5 CUDA prompt cache is opt-in.** With default settings
every turn reports `resumed at 0, kept 0` — cold and warm TTFT are identical
(0–6% delta), i.e. *zero* prefix reuse, while the 50.7 GiB prompt-cache budget
sits idle. The working opt-in for this family is **`TF_QWEN_REQUEST_STARTS=1`**
(the host store keeps a state per message start); the page/radix pair
`TF_QWEN_KV_SHARE`/`TF_QWEN_KV_PAGES` produced no observable reuse here and is
mutually exclusive with request-starts. See "The three cache modes" below.

This is a white-paper-grade result on its own: **multi-model serving inherits
per-family cache opt-ins**; a MoM orchestrator must set them per lane, or every
turn silently reprefills the whole conversation on both lanes.

**Metrics reported** (`experiments/results/CACHE-RESULTS.md`, raw
`cache-alternating.json`, aggregator `aggregate_cache.py`):
per lane and pass — TTFT p50/p95/max, e2e p50, decode tok/s (completion tokens /
decode seconds), total prompt tokens, prefill saved % (resumed / prompt tokens),
reuse turns, routed-correct count; plus cold→warm TTFT delta, final lane
residency/decisions, and routing failures (expected 0).

**Caveats.**
- `mom.memory` on `/v1/models` reports the **process** device ledger (one CUDA
  context on GB10 unified memory), identical for both lanes; it is not
  per-engine bytes. Per-lane weight bytes come from the B1 plan line instead.
- Decode tok/s uses server-side completion tokens over client-measured decode
  window (e2e − TTFT), so it includes stream framing overhead but not network.
- Pass 2 is "warm" in the same process; a cold-process baseline is the pass-1
  early turns themselves (history ~0), shown by the TTFT-vs-turn curve.

## The three cache modes (finding, verified on device)

`qwen3_5` CUDA has three mutually distinct prompt-cache behaviours; **none is
on by default**:

1. **Default (no env)** — host store exists and logs, but no marks are ever
   planned for this family: every request reports `resumed at 0, kept 0`,
   cold == warm TTFT, whole conversation reprefilled each turn.
2. **Page/radix mode** (`TF_QWEN_KV_SHARE=16 TF_QWEN_KV_PAGES=256`, the
   upstream "stable" profile pair) — pages go to the pool (visible in the
   startup line), but conversation reuse stayed at zero in our runs and the
   mode is **unobservable**: `lane_host.zig:453` reports only when the host
   store is wired, and radix mode nulls it (`own_prompt_cache`,
   `cuda_native.zig:216`). `cuda_prefill.zig:104` additionally requires
   `stream.chunks.len == 0` for adoption.
3. **Request-starts mode** (`TF_QWEN_REQUEST_STARTS=1`) — the family declares
   the switch (`cuda_native.zig:60`); the host store keeps a state at each
   message start and every turn after the first resumes from it. **This is
   the mode that delivers cache effectiveness for this family**, and it is
   mutually exclusive with mode 2 (`QwenRequestStartsWithSharedKv`).

White-paper implication: a multi-model orchestrator must set *and verify* the
cache mode per family — a MoM dashboard that only reads the host store will
silently display nothing while the engine (mode 2) does its own accounting.
`mom-serve` inherits the env, so each lane honors the mode; we verified
end-to-end with the engine's own `prompt cache:` lines.

## Results (TF_QWEN_REQUEST_STARTS=1, both lanes resident, whole run)

| pass | lane | n | TTFT p50 | TTFT p95 | TTFT max | e2e p50 | decode tok/s | prompt tokens | prefill saved | reuse turns | routed ok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 cold | qwen3.5-2b | 14 | 46 ms | 48 ms | 48 ms | 0.23 s | 171.0 | 19189 | **85.4%** | 13/14 | 14/14 |
| 1 cold | qwen3.6-35b | 14 | 177 ms | 186 ms | 186 ms | 2.21 s | 62.9 | 20065 | **85.7%** | 13/14 | 14/14 |
| 2 warm | qwen3.5-2b | 14 | 11 ms | 12 ms | 12 ms | 0.20 s | 171.1 | 19189 | **99.5%** | 14/14 | 14/14 |
| 2 warm | qwen3.6-35b | 14 | 28 ms | 30 ms | 30 ms | 2.06 s | 63.0 | 20065 | **99.5%** | 14/14 | 14/14 |

- Cold→warm TTFT: trivial 46→10 ms (**77% faster**), work 175→28 ms (**84% faster**).
- Even on the cold pass, turn k+2 resumes from turn k's state (85% saved):
  alternating traffic never loses more than its two interleaved turns.
- Engine-side proof: `resumed at 2708` of 2715 tokens late in the conversation,
  cumulative hits 27 per lane, 14–15 kept states (484 MiB small / 1311 MiB big).
- Decode throughput is cache-independent by design: 171 tok/s (2B), 63 tok/s
  (35B NVFP4 + MTP) across all passes.
- Routing: **28/28 per pass, 0 failures** (`--no-sticky`, content-routed).
- Residency: both lanes `resident` throughout (no sleep), process ledger 27.9 GiB.
- Cache-effectiveness gate: 54/56 reuse opportunities hit (the 2 misses are the
  two turn-1s, nothing to reuse — by construction).
- Raw data: `experiments/results/cache-alternating.json`; tables regenerate
  via `python3 aggregate_cache.py`.
