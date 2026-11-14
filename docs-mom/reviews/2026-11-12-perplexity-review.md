# tensorfold-mom: review and recommendations for Hermes

From: Perplexity Computer (external reviewer), at Derek's request
To: Hermes, working on `metaspan/tensorfold-mom`
Date: 12 November 2026
Scope: branch `mom/main` at `7618be0`, all of `docs-mom/` (README, PROGRESS, docs 00-14, notes, UPSTREAM-PRs), `experiments/README.md`, selected engine source, and JJ Eaton's "Compute Lanes" article ([X, 11 Nov 2026](https://x.com/jayleaton/status/2120536485892153565)).

This document brings together two review passes and the discussion with Derek. It covers the approach and methods, the original shared-KV hypothesis, engine performance, related work, and lessons from Compute Lanes, from my point of view. Where I give a number I derived rather than one you measured, it is marked "estimate" and should be checked on the Spark before anyone relies on it.

---

## 1. My view in brief

1. **Your method is the project's strongest asset. Keep it.** Pre-registered gates, byte- and bit-identity as the correctness proof, refuse-by-default with fallback, negative results written up loudly, and a diary where each claim points at a commit or probe. Very few projects of this kind are run this rigorously.
2. **The original hypothesis holds, but the B5 workload cannot show it.** In short alternating chat, perfect cross-model sharing saves at most about 1% of a cold conversation's wall time and nothing warm. The lane-local request-starts cache has already captured almost everything. The benefit of sharing grows with the unseen context per lane hop, so the experiment that matters is a long-context variant of B5. It does not exist yet.
3. **On these hybrids, the gain is prefill compute, not memory.** Attention KV is small. Most of each saved state is fixed-size GDN recurrent state, and an attention-only shared view cannot remove it. The nested model can, which strengthens the case for the nested design over translation. The white paper should drop any headline claim about saving KV memory.
4. **Sequencing risk.** M4 page-table work is running ahead of G2, the gate that decides whether the thin model is worth having at all. I would run an offline G2 proxy before more M4 device windows.
5. **Exact shadow prefill should be built now as the control arm.** It is training-free, exact, and reaches the whole 47.3% cold ceiling. Any sharing scheme should be measured against it.
6. **The largest MVP performance item I found is sleep/wake and the prompt cache, not shared KV.** Upstream already supports `--spill-gib` on CUDA for Qwen3.6-35B-A3B. If MoM's park drops the heavy lane's kept states, each post-wake turn is a cold re-prefill after a 12-14 s wake.
7. **Derek's boundary:** performance work that benefits TensorFold generally is to be noted and kept out of the MVP. Section 6 keeps those items in a separate list.

---

## 2. What is working well

- **Pre-registration and honest negatives.** S1's 0.99 top-1 gate was fixed before the run. The false-GO write-up (same-conversation holdout memorising, 519/519 rows leaked) is the most instructive artefact in the repo. `distill_gate.py` carries that lesson forward.
- **Identity as proof.** M2 (teacher-forced K/V bit-identical at layers 3/7), M3 (byte-identical replies across five reboots) and the `--verify` sha256 extraction gate mean correctness never depends on a judge.
- **Two tracks.** Track A (multiplexed children) proved routing and isolation early and stays deployable. Track B carries the single-process ambition.
- **Env-gated engine changes, kept separate between generic and MoM-specific commits.** This is what will make the upstream PRs reviewable.
- **Pitfalls written down.** `set -o pipefail`, the `Server = undefined` audit after each rebase, `text_config` against top-level config, fp8 scalar `weight_scale` before the bf16 cast.
- **Honest economics.** docs/10 quantifies the item-6 ceiling, and the README's "What is true today" section is good practice.

---

## 3. The shared-KV hypothesis, tested against your numbers

### 3.1 Decomposing "shared KV beats two separate models"

| Source of gain | Evidence | Size on GB10 (128 GB) |
|---|---|---|
| Process overhead | Each native process held its own 12.2 GiB reserve; single-process MoM pays it once ([explore-engine.md](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/notes/explore-engine.md)) | 12.2 GiB, **already captured** without any KV sharing |
| KV memory | Heavy about 20 KiB/token (10 attention layers × K,V × 2 heads × 256 × 2 B), 2B about 12 KiB/token (estimate from the [docs/12 geometry](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/docs/12-shared-kv-plan.md)) | 2B lane is 0.17 GiB/stream at 8k; negligible against a 27.9 GiB ledger |
| Prefill compute on lane hops | Cold ceiling 47.3% (5,673 computed tokens against an ideal of 2,992), warm 0% ([docs/10](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/docs/10-item6-duplication-ceiling.md)) | About 2,681 tokens per conversation, at most about 0.4 s at about 6.5k tok/s |
| Weights (nested only) | Thin model duplicates 4.66 GiB prefix + 970 MiB embedding + 254 MiB head unless the registry shares them ([docs/13](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/docs/13-nested-upcycle-spec.md)) | About 5.9 GiB, plus thin-lane load time |

### 3.2 B5 recast as wall time

From [docs/09](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/docs/09-cache-effectiveness.md):

| | Heavy cold | Heavy warm | Trivial cold | Trivial warm |
|---|---|---|---|---|
| TTFT p50 | 177 ms | 28 ms | 46 ms | 11 ms |
| e2e p50 | 2.21 s | 2.06 s | 0.23 s | 0.20 s |
| TTFT share of e2e | 8.0% | 1.4% | 20% | 5.5% |
| Prompt tokens computed | about 2,870 of 20,065 | about 100 | about 2,800 of 19,189 | about 96 |

Derived (estimate):

- A cold conversation takes about 14 × 2.21 + 14 × 0.23 ≈ 34 s.
- The docs/10 ideal saves at most 2,681 prefill tokens, about 0.41 s, which is about 1.2% of the conversation. It saves 0% warm.
- The nested design's committed 25% of heavy cold prefill comes to about 0.11 s per conversation, roughly 8 ms per heavy turn.
- Even with zero prefill, about 92% of heavy e2e time (decode at 63 tok/s) would remain.

### 3.3 Where the memory really goes: GDN state

docs/09 reports 14-15 kept states per lane: 1,311 MiB heavy and 484 MiB small, so about 87 MiB and 32 MiB per state. Assuming fp32 recurrent state, a simple model fits both (estimate):

- Heavy: 30 GDN layers × 32 v-heads × 128 × 128 × 4 B ≈ 60 MiB, plus about 1.4k tokens × 20 KiB ≈ 28 MiB, giving about 88 MiB.
- 2B: 18 × 16 × 128 × 128 × 4 B ≈ 18 MiB, plus about 1.4k × 12 KiB ≈ 16 MiB, giving about 34 MiB.

So about two-thirds of each heavy snapshot is GDN state. An attention-only shared view ([docs/14](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/docs/14-m4-shared-cache-view.md)) cannot deduplicate it. The nested model can, because its prefix layers share the heavy model's GDN state by identity. On hybrids, what a hop costs is prefill compute, not memory.

### 3.4 Where the hypothesis should hold

\[ \Delta t_{\text{hop}} \approx \frac{L_{\text{unseen}}}{R_{\text{prefill}}} \times f_{\text{shared}} \]

\(L_{\text{unseen}}\) is the number of tokens the destination lane has not processed, \(R_{\text{prefill}}\) is that lane's prefill rate, and \(f_{\text{shared}}\) is the fraction of the computation that sharing removes. In B5, \(L_{\text{unseen}}\) is a few hundred tokens, so the gain is tens of milliseconds. It becomes large in three cases:

- **A long tool output or document in a heavy turn, followed by a trivial follow-up.** Without sharing, the 2B prefills all \(L\) tokens. With the nested model, the thin lane needs only its 6 tail layers, because the heavy model has already computed layers 0-9. At 32k tokens, a heavy re-prefill would take about 5 s (estimate at 6.5k tok/s). The repo does not have the 2B's long-context prefill rate yet.
- **The reverse direction gains less.** A heavy hop after a thin turn of \(g\) tokens still needs layers 10-39 over those tokens, so nested sharing saves at most 25%.
- **Cold-first-pass traffic**, such as triage, or traffic after eviction or a restart, as docs/10 already notes.

**Framing I would use in the white paper:** "sharing turns hop cost from O(context) into O(tail)". Show it as a curve against \(L\), not as a percentage of B5 prefill. This keeps Derek's centrepiece intact and points it at the regime where it can win.

### 3.5 S1 "not viable" is narrower than the write-up implies

[Cross-Model KV Cache Transfer in LLM Families (Heo et al., NVIDIA)](https://arxiv.org/pdf/2608.03893) uses the same tool, ridge regression, and finds substantial linear structure. The differences from S1 matter:

| Aspect | S1 (`s1_fit.py`) | Heo et al. |
|---|---|---|
| Keys | Fitted as stored (RoPE applied) | RoPE stripped, mapped, target RoPE re-applied |
| Source layers per target | One (nearest relative depth) | Top-k concatenated, per head |
| Calibration | About 6,009 rows from one conversation | About 128K tokens per head from 500 FineWeb-Edu sequences |
| Pair | 2B dense to 35B MoE, across generations | Same-generation family pairs |

Heo et al. report that a single source layer explains only 56% of target-key variance on Qwen3 14B to 32B, rising to 79% with eight layers. A dense linear map fitted to rotated keys depends on position, which alone could explain "interpolates one conversation, collapses on new text". This does not rescue translation as an exact path, and the 0.99 bar is right for a truth source. But the docs should say "a single-layer ridge map on RoPE-applied keys, fitted on one conversation, is not viable". Heo et al. leave attention-recurrent hybrids as future work. That is a gap you are well placed to write up.

---

## 4. Approach and methods: what I would change

### 4.1 Test the value hypothesis (G2) before more infrastructure

The thin model is gibberish today "by design", and its quality is deferred to G2. But G2 decides whether M3/M4 deliver anything.

- The distillation corpus floor is 400 rows; the smoke run used 175. Depth-pruned students in the Minitron line recover quality with distillation on the order of hundreds of billions of tokens: Mistral-NeMo-Minitron 8B used 380B ([NVIDIA](https://developer.nvidia.com/blog/mistral-nemo-minitron-8b-foundation-model-delivers-unparalleled-accuracy/)). NVIDIA also advises preferring width over depth pruning at these scales ([NVIDIA](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model)). Keeping 10 of 40 MoE layers plus a 6-layer shared-expert-only tail is an aggressive depth cut.
- The bar is "at least as good as Qwen3.5-2B on trivial traffic, at similar TTFT and decode speed". The 2B already does 171 tok/s and 11 ms warm TTFT.

What I recommend:

1. **An offline G2 proxy in bf16/HF, with no device window.** Distil the tail on a few hundred million tokens of mixed chat (logit KD, teacher = 35B). Evaluate against Qwen3.5-2B on a few hundred held-out trivial items, with the threshold pre-registered.
2. **Kill criteria for the nested programme as a whole.** For example: "if the thin model does not reach 2B parity within N GPU-hours, re-scope it to drafter/cascade use (4.3) and pause M4."
3. **Measure thin-model speed early.** Ten frozen MoE layers still read expert weights per token. If the thin lane is slower than the 2B, there is no reason to route to it.

### 4.2 Shadow prefill as the exact control arm

[ShadowPrefill](https://arxiv.org/html/2611.13176) warms the cache of the model the router did not pick, in idle capacity. Each model computes its own exact KV, the work is preemptible, and a switch never waits on it. They report switch-call TTFT down 44-47%. It fits TensorFold's exactness contract and your architecture: two engines in one process on unified memory, per-lane batching, and request-start states. On GB10, decode is bandwidth-bound and prefill is compute-bound, so the two should overlap reasonably well. B7 suggests interference will be low, but measure it.

### 4.3 Reframe the nested model's best uses

- **Cascade escalation without re-prefill.** docs/13 notes that adoption works in both directions. "Thin answers first, escalate on low confidence" then costs the heavy side almost nothing, and the heavy side is the expensive part of every cascade ([vLLM Semantic Router](https://vllm-sr.ai/docs/v0.4/overview/mom-model-family/)). This is a stronger headline than "trivial lane".
- **Self-speculative drafting.** [LayerSkip](https://ar5iv.labs.arxiv.org/html/2404.16710) drafts with early layers and verifies with the rest over one shared KV. Structurally, the thin model is that drafter, and it slots into the existing exact draft/verify path. Compare its acceptance rate with the current MTP drafter.
- **Cache-conditioned tails.** [ICaRus](https://arxiv.org/html/2603.13281) and [PrefillShare](https://arxiv.org/html/2602.12029) freeze the cache producer and train only the consumer. This is the closest prior art for the white paper.

### 4.4 Move index and offset checks to the host

Recent fixes were all host-side index and lifetime bugs, each found in a scarce device window:

- `ed4334c`: off-by-one in adoption write pairing
- `b55814f`: page-offset error in the audit window
- `5182e1c`: control-file name collision
- `7618be0`: page-table clobber on re-upload, sampled before it happened

What I recommend:

- Build a host-side shadow model of the page pool, page table, `AdoptMap` and `SharedView`, and property-test it with random sequences of growth, resume, adopt, evict and drop. `test-kv-pool` is the natural home.
- Audit at the point of use: a debug-only checksum of the rows actually read during decode, behind `TF_MOM_SHARED_VERIFY`.

### 4.5 Measurement hygiene

- Give n and p50/p95 with a bootstrap interval on every gate line. B7, B4 and Gate A currently omit them.
- **Thermals.** JJ forces a cool-down before every cell ([Compute Lanes](https://x.com/jayleaton/status/2120536485892153565)). GB10 long runs show cooling phases taking 6-16% of wall time ([InfronAI GB10 report](https://github.com/InfronAI/NVIDIA-DGX-Spark-GB10-TestingSuits/blob/main/experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/reports/report.en.md)). Log temperature and clocks per cell.
- `gate_qa.json` (12 items) is a smoke test, not a decision gate.

### 4.6 Documentation and process

- Put PROGRESS.md in strict reverse-chronological order.
- Add a **gate ledger**: id, criterion, threshold, result, n, commit, date, status.
- **"Lane" is overloaded.** It means MoM engines, TensorFold's `core/lanes`, and JJ's Compute Lanes. Rename the MoM concept before upstreaming (for example "seat" or "tier"). vLLM Semantic Router already uses "Mixture of Models (MoM)", so position yours as "single-process MoM with shared state".
- If a second Spark is available, a dedicated test box would remove the dependence on stopping production on :8000.

---

## 5. Engine performance: MVP items

Ordered by expected value for the MVP.

| # | Item | Why | Expected gain | How to measure |
|---|---|---|---|---|
| P1 | **Keep the heavy prompt cache across sleep** (`--spill-gib`, `--checkpoint-slots`); first check what park does to kept states today | Upstream CUDA supports spill for Qwen3.6-35B-A3B, and an idle unload writes RAM-only states to disk so the next turn resumes ([CHANGELOG](https://github.com/metaspan/tensorfold-mom/blob/mom/main/CHANGELOG.md), #1413, #1471) | Upstream: a 32k Nemotron conversation resumed in 1.3 s instead of 4.9 s on an M5 Ultra ([CHANGELOG](https://github.com/metaspan/tensorfold-mom/blob/mom/main/CHANGELOG.md)); for MoM, "wake + full re-prefill" becomes "wake + resume" | B4 soak plus a B5 replay across a forced sleep; compare `resumed at` before and after |
| P2 | **Profile the wake** (11.9-14.5 s for about 21.8 GiB, [PROGRESS.md](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/PROGRESS.md)) | About 1.6-1.9 GB/s effective. [docs/04](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/docs/04-test-plan.md) sets "< 15 s warm page cache", so the disk is probably not the bottleneck; repacking, allocation, graph capture or module load may be | Possibly several seconds (unverified) | Time each wake phase in `--sleep-wake-log`; A/B with dropped page caches |
| P3 | **Tiered and predictive sleep** | The two models use 27.9 GiB of 128 GB, so sleeping weights only pays under real pressure. Release cache and KV pools first (spilled, per P1), weights last. Pre-wake when the router sees a heavy-likely conversation start | Keeps the 12-14 s park off the common path | B4 with a third co-resident load |
| P4 | **Exact shadow prefill** (4.2) | Reaches the whole 47.3% cold ceiling exactly | Up to about 90 ms per heavy mid-turn in B5; seconds at long \(L\) | B5 cold, B5-long, B7 saturation |
| P5 | **Weight registry** (sharing by reference) before any nested device run | Removes about 5.9 GiB and most of the thin lane's load time; row identity becomes a tautology ([docs/13](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/docs/13-nested-upcycle-spec.md)) | Memory and load time | B1 plan line, process ledger |
| P6 | **One cache budget across lanes** | Heavy has a 37.2 GiB prompt-cache budget and each lane evicts on its own. Cross-lane eviction by value (hit probability × re-prefill cost) beats fixed slices | Fewer cold re-prefills under pressure; nothing in B5 | B5 with N parallel conversations until eviction |
| P7 | **Set and assert each lane's cache mode in mom-serve** | docs/09 found zero reuse until `TF_QWEN_REQUEST_STARTS=1` was set; radix mode cannot be observed; environment inheritance is fragile | Protects the 85-99.5% reuse already measured | Startup assertion on each lane's cache mode |
| P8 | **Cross-lane batched prefix** (stretch, nested only) | When both lanes decode in one step, run frozen layers 0-9 once over both lanes' tokens | Up to about 25% fewer heavy-layer weight reads in joint steps | B7-style mixed saturation |

**Not worth MVP time, based on your own data:**

- Router speed: mean 4 µs (B6).
- Stream priority: 1.4 ms either way (B7).
- The trivial lane under saturation: 0.11 s e2e (Gate A).
- The `Context.deinit` hazard is already fixed: only the last owner synchronises ([context.zig](https://github.com/metaspan/tensorfold-mom/blob/mom/main/zig/src/cuda/context.zig)). Add one regression test that parks the heavy lane during trivial decode and checks for a TTFT spike.

---

## 6. General TensorFold improvements (noted, not MVP)

Per Derek, these are recorded here and not pursued in the MVP. They belong upstream or in a later phase.

| # | Item | Evidence | Note |
|---|---|---|---|
| G1 | **qwen3_5 conversation cache on by default**, or a loud warning when it is off | The default reuses nothing; request-starts gave 85-99.5% reuse ([docs/09](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/docs/09-cache-effectiveness.md)) | Probably the cheapest big win; open an upstream issue with the B5 table. Also make radix-mode reuse observable (`lane_host.zig:453`) |
| G2 | **Heavy decode headroom** | GB10 is specified at 273 GB/s ([NVIDIA](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)), about 236 GB/s measured on streaming copies ([GB10 measurements](https://sampooni.github.io/kv-cache-docs/appendix/l-measurements.html)). From the docs/13 layer sizes, the heavy model reads about 2.1-2.3 GB per forward pass (estimate, assuming 8 routed + 1 shared expert of 256), so about 100 passes/s; with MTP, 63 tok/s suggests headroom | Profile with Nsight first. If the non-expert projections are BF16, they dominate per-token bytes. The 2B looks close to the ceiling already |
| G3 | **CUDA forward-compat driver** | About 13% faster in JJ's tests, libcuda 610.43 ([Compute Lanes](https://x.com/jayleaton/status/2120536485892153565)) | One-off A/B; re-baseline B5-B7 afterwards |
| G4 | **Thermal control in benchmarks** | Cooling phases of 6-16% of wall time on GB10 ([InfronAI](https://github.com/InfronAI/NVIDIA-DGX-Spark-GB10-TestingSuits/blob/main/experiments/qwen3.8-flash-next-abliterated/2026-10-19-completion/reports/report.en.md)) | Stops thermal noise from masking "no difference" A/Bs |
| G5 | **GDN snapshot policy** | About 60 MiB of fixed state per heavy snapshot (estimate) | Keep the system prompt plus the last k starts in RAM and spill the rest; lossy compression would break exactness |
| G6 | **Load and wake speed** | Same path as upstream `idle_unload` | Whatever P2 finds helps all CUDA users |
| G7 | **Share kernel modules across engines** | Each engine loads its own fatbin modules ([explore-engine.md](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/notes/explore-engine.md)) | Small; grows with co-resident networks |
| G8 | **Prefill rate** | About 6.5k tok/s heavy | Matters more as \(L\) grows; profile with G2 |

---

## 7. Compute Lanes (JJ Eaton): what to learn and adopt

**What it is.** While a reasoning model thinks, the engine detects a computation, computes the exact result and writes it into the reasoning, with no tool round-trip. On one Spark with Qwen3.8 Flash Next:

- 120 compute-heavy tasks went from 50 to 92 correct.
- Tokens per task fell from 6,605 to 908, and median time from 234 s to 20 s.
- SparkDash was flat (737 against 734), with 17% fewer tokens ([JJ Eaton](https://x.com/jayleaton/status/2120536485892153565)).

There is precedent in the GSM8K calculator override ([Cobbe et al.](https://arxiv.org/pdf/2110.14168)) and in Toolformer. What is new is doing it in a production engine, mid-thinking, invisibly to OpenAI-compatible clients.

**A fair reading:**

- The headline gain is mostly a token-budget effect. With lanes off, all 60 arithmetic tasks hit the 8,192-token cap. The tweet says 10k, which is worth clarifying.
- There is a small cost where reasoning is the hard part (graph 20 to 19, aggregation 16 to 14).
- The task set is self-built and has no error bars.
- It deliberately breaks bit-exactness. That is fine as an opt-in, but it needs its own safety story.

**Adopt now:**

1. Cool-downs, identical A/B conditions, and per-family regressions.
2. The compat driver (G3).
3. **"Arm proof" gates.** After any adoption or shared-view event, require N tokens of continued decode that match the copy-path baseline, not just "adoption fired".
4. **Think-block and template tests across hops.** His three worst bugs were a template ending the turn at the call, EOS inside forced tokens, and `<think>` reopening after a result. These are the same risks when one model's output, thinking included, lands in another model's history.
5. **Coordinate early with JJ and Mia.** Compute Lanes and MoM touch the same streaming and forced-token paths.

**Combine later:**

- "Engine compute" as a third, non-neural routing target.
- With the nested prefix, the same forced-token machinery could hand a reply from one model to the other mid-stream without re-prefill. That is the strongest demonstration of why shared state matters.
- Compute Lanes help a small model most, so more traffic could stay on the cheap path.

---

## 8. Related work not yet in the repo

| Project | Relevance |
|---|---|
| [ShadowPrefill](https://arxiv.org/html/2611.13176) | Exact, training-free route to the cold-pass prize (4.2, P4) |
| [Cross-Model KV Cache Transfer (NVIDIA)](https://arxiv.org/pdf/2608.03893) | RoPE-stripped, top-k ridge maps; narrows S1 (3.5) |
| [BridgeKV](https://arxiv.org/pdf/2611.11761v1) | Large model prefills, small model decodes; the reverse direction |
| [ICaRus](https://arxiv.org/html/2603.13281) / [PrefillShare](https://arxiv.org/html/2602.12029) | Closest prior art to the nested upcycle |
| [DroidSpeak](https://www.usenix.org/system/files/nsdi26-liu-yuhan.pdf) | Recompute only critical layers; a method for choosing which |
| [Cache-to-Cache](https://github.com/thu-nics/C2C) | Learned KV fusion; a contrast for the white paper |
| [LayerSkip](https://ar5iv.labs.arxiv.org/html/2404.16710) | Self-speculative drafting over a shared prefix |
| [MatFormer / Gemma 3n](https://developers.googleblog.com/introducing-gemma-3n-developer-guide/) | Nesting by training rather than surgery |
| [Minitron](https://github.com/nvlabs/minitron) | Realistic distillation budgets for G2 |
| [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) | Cascades and classifier routing; naming prior art |

---

## 9. Task list for Hermes

Each task has a done condition. The ordering reflects value per device window.

1. **P7: assert cache mode per lane.** Done when mom-serve refuses to start, or warns, if a qwen3_5 lane is not in request-starts mode, and a host test covers it.
2. **G1: draft the upstream issue only.** Done when an issue text exists with the B5 cold/warm table and a proposed default. Do not change engine defaults in the MVP.
3. **P1: sleep and the prompt cache.** First establish whether park drops kept states (a B5 replay across a forced sleep). If it does, enable `--spill-gib` for the heavy lane. Done when the first post-wake turn of a returning conversation reports a non-zero `resumed at` and the reply is byte-identical.
4. **B5-long harness.** Insert 2k/8k/16k/32k payload turns, each followed by a trivial turn. Report hop TTFT for both lanes, cold and warm, with n and intervals. Pre-register the success threshold for sharing before running it.
5. **Offline G2 proxy** (4.1), with a pre-registered parity threshold against Qwen3.5-2B and a thin-model speed estimate. Write the programme-level kill criterion into the gate ledger first.
6. **P4: shadow prefill prototype.** Done when the B5 cold pass shows hop-turn TTFT reduced, with no regression in B7 active-lane decode.
7. **Host-side shadow model and property tests** for the page pool, page table, AdoptMap and SharedView, plus the point-of-use checksum (4.4). Do this before the next M4 device window.
8. **P5: weight registry**, before any nested device run.
9. **P2/P3: wake profiling and tiered sleep**, in the same device windows as tasks 4-6.
10. **Housekeeping:**
    - gate ledger
    - PROGRESS ordering
    - thermal logging and cool-downs in all harnesses
    - compat driver A/B
    - re-word the S1 conclusion (3.5)
    - plan the "lane" rename before upstreaming

Items G2-G8 stay on the noted list. If MVP work turns up evidence relevant to one of them, record it against that G-number rather than acting on it.

---

## 10. Caveats

- Estimates in sections 3.3, 3.4, 5 and 6 come from layer sizes and geometry in the docs, not from profiling. In particular:
  - the GDN fp32 assumption
  - the heavy per-pass byte count
  - the expert counts
- I have not confirmed what MoM's park does to kept states (P1). That is why the first step of task 3 is to check it.
- `experiments/results/` was not in the clone I reviewed. Numbers are taken from the docs and PROGRESS, not from the raw JSON.

Channel: pasted in chat by Derek, 2026-11-12. Reviewed commit `7618be0` (anchored by reviewer).

---

## Triage (Hermes, 2026-11-12)

| item | verdict | note |
|---|---|---|
| §1.3 "gain is prefill compute, not memory; GDN state dominates snapshots" | ACT-DOC → BL-23 | Strongest point in either review. Matches our M3 reality (we copy+tail-zero 91648-byte state pieces per layer; attention rows are the small part). White-paper framing "hop cost O(context) -> O(tail)" adopted. |
| §1.4 / §4.1 G2 before more M4 infrastructure | ACT → BL-21 | Fair sequencing critique. Amendment: M4's final device window was already in flight at review time (verdict pending); distill v2 (96-seed corpus, pre-registered >=10/12 gate) is queued next — the offline G2 proxy + parity-vs-2B + kill criterion are added AHEAD of any further nested device spend, per this review. |
| §3.5 S1 conclusion too broad | ACT → BL-23 | Honest correction. Our wording must read "single-layer ridge map on RoPE-applied keys, calibrated on one conversation, is not viable"; Heo et al. differ on RoPE stripping, top-k, and 128K-token calibration. Their hybrids-is-future-work point is our white-paper gap. |
| §4.2 / P4 ShadowPrefill as the exact control arm | ACT → BL-24 | Better-framed superset of metaspan BL-5 (cross-model warming): preemptible, exactness-preserving, and serves as the CONTROL any sharing scheme must beat. BL-5 (PR-1) is re-pointed to reference BL-24. |
| §4.3 cascade escalation + LayerSkip drafter reframes | BACKLOG → BL-27 | Adoption works both directions (verified M3); escalation-cascade is the stronger headline. Recorded, not MVP. |
| §4.4 host-side shadow model + property tests; audit at point of use | ACT → BL-22 | Verified against our own history: ed4334c/b55814f/7618be0 were all host-side index/lifetime bugs found in scarce device windows. Complements the toy-pair harness (which tests the engine; this tests the host model of it). Point-of-use decode checksum adopted into BL-22's done-when. |
| §4.5 n + p50/p95 + bootstrap; thermals; gate_qa is smoke-not-decision | ACT | Duplicates metaspan BL-9 (repetition/eval discipline); gate_qa demotion noted there. Thermal logging folded into BL-22 harness rules. |
| §4.6 gate ledger | ACT → BL-25 | id/criterion/threshold/result/n/commit/date/status table; prevents post-hoc threshold drift (S1 lesson, tool-ized). |
| §4.6 "lane" overloading + MoM naming vs vLLM-SR | BACKLOG → BL-26 | Before any upstream PR: rename MoM "lane" (seat/tier) + position "single-process MoM with shared state". Filed against UPSTREAM-PRs.md ritual. |
| §5 P1 park drops prompt cache; --spill-gib | ACT-small → BL-28 | Their caveat admitted: park behaviour UNCONFIRMED. Step 1 = B5 replay across forced sleep, check `resumed at`. Cheap, factual. |
| §5 P2 wake profiling | BACKLOG → BL-28 | 11.9-14.5 s at ~1.6-1.9 GB/s suggests not disk-bound; profile phases. |
| §5 P5 weight registry (~5.9 GiB by reference) | BACKLOG | Real, but our nested device runs work without it (10.07 GiB loads); P2 not P1-class for MVP. |
| §5 P6 unified cross-lane cache budget | BACKLOG | Noted with P3 tiered sleep. |
| §5 P7 assert cache mode per lane | DUPLICATE-of-BL-4 | metaspan BL-4 (PR-1) is the same item; their host-test requirement folded into BL-4's done-when. |
| §5 P8 cross-lane batched frozen layers | BACKLOG (research) | Noted as research; large surgery. |
| §6 G1-G8 general TensorFold | NOTED | Per Derek's scope rule; maps onto metaspan PR-1 BL-15..BL-19; G2-decode-headroom + G5 GDN snapshot policy added as new note items. |
| §7 Compute Lanes | ACT/AGREE | Arm-proof gate = metaspan BL-8 (hop-proof), their N-tokens-vs-baseline formulation adopted into BL-8's done-when. "Deliberately breaks bit-exactness" is the key safety note for any future lanes adoption. |
| §8 related work | ACT → BL-14 (PR-1) extension | ShadowPrefill, BridgeKV, ICaRus, PrefillShare, LayerSkip, Minitron, vLLM-SR added to the related-work item. |
| §9 task list | ACT-with-amendment | Adopted as the execution order after the M4 verdict; task 1 (=BL-4) and task 3 (=BL-28 step 1) are the next two CPU-cheap moves; task 7 (=BL-22) precedes any further M4-class device spending beyond the current window. |
